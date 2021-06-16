import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np
import mlflow
import shutil

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

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
                if random.uniform(0,1) > self.p:
                    i = random.randint(0, self.max_size-1)
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
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

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
        self.started= False

    def update_average(self, old, new):
        if not self.started:
            self.started= True
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
