import datetime
import random
import shutil
import sys
import time

import cv2
import mlflow
import numpy as np
import pylibjpeg
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps
from pydicom import dcmread
from torch.autograd import Variable
from tqdm import tqdm
from skimage.filters import hessian,meijering


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def tophat(img, fsize=20):
    img = np.array(img).astype(np.float32)
    filterSize = (fsize, fsize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    wth = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    bth = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    dst = (img + wth - bth).clip(0, 255).astype(np.uint8)
    return T.ToPILImage()(dst)

def hessian_enhance(img, sigma=[0.1]):
    img = np.array(img)
    dst = hessian(img, sigmas=sigma, black_ridges=True)
    dst = (dst*255).astype('uint8')
    return T.ToPILImage()(dst)

def meijering_enhance(img, sigma=[0.1]):
    img = np.array(img)
    dst = meijering(img, sigmas=sigma, black_ridges=True)
    dst = (dst*255).astype('uint8')
    return T.ToPILImage()(dst)


class logger():
    def __init__(self, exp_name, run_name):
        mlflow.set_experiment(opt.exp_name)
        run = mlflow.start_run(run_name=opt.name)
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        run_dir = f'mlruns/{experiment_id}/{run_id}'
        art_dir = f"{run_dir}/artifacts"
        ckpt_path = f"{run_dir}/last.ckpt"
        mlflow.log_params(vars(opt))
        source_code = [i for i in os.listdir() if ".py" in i]
        for i in source_code:
            shutil.copy(i, f"{art_dir}/{i}")


class ReplayBuffer():
    def __init__(self, max_size=50, p=0.5):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
        self.p = p

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > self.p:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.started = False

    def update_average(self, old, new):
        if not self.started:
            self.started = True
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


def fusion_predict(model, ckpts, x, size=256, pad=0, device='cpu', return_x=True, multiangle=True, denoise=3, cutoff=1):
    """融合多个角度或多个ckpt的输出,可取得更好的结果
    ckpt：checkpoint path
    x：input tensor, shape(C,H,W)
    pad：边缘填充, 如果mutiangle=False, 仅填充right和bottom，如果multiangel=True, 填充四边。
         pad可增加边缘血管的提取，但也可能增加噪声
    return_x: 是否返回转换成图片的x
    multiangle: 是否多角度预测
    denoise: 去噪强度
    """
    outs = []
    B0 = x.to(device)
    if multiangle:
        B1 = T.functional.rotate(B0, 90)
        B2 = T.functional.rotate(B0, 180)
        B3 = T.functional.rotate(B0, 270)
        B = torch.stack((B0, B1, B2, B3))
        B = T.functional.pad(B, pad, padding_mode='reflect')  # 仅pad了底边
    else:
        B = B0.unsqueeze(0)
        B = T.functional.pad(B, (0, 0, pad, pad), padding_mode='reflect')  # 仅pad了底边
    if return_x:
        B_dnorm = torchvision.utils.make_grid(B0, normalize=True, padding=0)
        B_dnorm = T.ToPILImage()(B_dnorm)
        B_dnorm = T.CenterCrop(size)(B_dnorm)

    for ckpt in ckpts:
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['netE'])
        model.to(device)
        with torch.no_grad():
            fakeA = model.model(B)
        if multiangle:
            fakeA[1] = T.functional.rotate(fakeA[1], 270)
            fakeA[2] = T.functional.rotate(fakeA[2], 180)
            fakeA[3] = T.functional.rotate(fakeA[3], 90)
            fakeA = T.CenterCrop(size)(fakeA)
        else:
            fakeA = fakeA[:, :, :size, :size]

        outs.append(fakeA)

    out = torch.cat(outs)
    out = out.max(0, True)[0]
    out = torchvision.utils.make_grid(out, normalize=True)

    out = (out.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
    out = cv2.fastNlMeansDenoising(out, None, denoise, 7, 21)

    out = T.ToPILImage()(out)
    out = ImageOps.autocontrast(out, cutoff=cutoff)

    if return_x:
        return B_dnorm, out
    else:
        return out


def make_gif_from_dicom(src, dst, model, ckpts, pad=0, device='cpu', multiangle=True, denoise=5, cutoff=1):
    """读取dicom，提取血管后转换为gif图片
    scr: dicom地址
    dst: 输出gif地址
    model: 输入模型nn
    ckpts: 模型的checkpoint，list，可以多个
    pad：边缘填充, 如果mutiangle=False, 仅填充right和bottom，如果multiangel=True, 填充四边
    device: 设备号，比如'cpu','cuda:0'
    输出: gif
    """
    arr = dcmread(src).pixel_array
    tsr = torch.from_numpy(arr) / 255
    tfmc = T.Compose([
        T.Resize(256),
        T.Normalize((0.5,), (0.5,))
    ])
    tsr = tfmc(tsr)
    imgs = []
    for i in tqdm(range(tsr.shape[0])):
        B = tsr[i].unsqueeze(0)
        B_dnorm, fakeA = fusion_predict(model, ckpts, B, pad=pad, device=device, multiangle=multiangle, denoise=denoise, cutoff=cutoff)
        B_dnorm = T.ToTensor()(B_dnorm)
        fakeA = T.ToTensor()(fakeA)

        grid = torch.stack((B_dnorm, fakeA), dim=0)
        grid = torchvision.utils.make_grid(grid)
        img = T.ToPILImage()(grid)

        imgs.append(img)
    img.save(dst, save_all=True, append_images=imgs)
