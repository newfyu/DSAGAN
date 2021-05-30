import argparse
import itertools
import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from datasets import ImageDataset
from models import Discriminator, Generator
from utils import LambdaLR, Logger, ReplayBuffer, weights_init_normal

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/cycledsa_v2/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--device', type=str, default='cpu', help='select device, such as cpu,cuda:0')
parser.add_argument('--log_step', type=int, default=100, help='select device, such as cpu,cuda:0')
parser.add_argument('--exp_name', type=str, help='trial name', default='Default')
parser.add_argument('--name', type=str, help='trial name', required=True)
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='resume train from checkpoint')

opt = parser.parse_args()
print(opt)

device = torch.device(opt.device)
###### Definition of variables ######
# Networks
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)

if opt.device != 'cpu':
    netG_B2A.to(device)
    netD_A.to(device)

netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_B2A.parameters(),lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Set logger
mlflow.set_experiment(opt.exp_name)
run = mlflow.start_run(run_name=opt.name)
run_id = run.info.run_id
experiment_id = run.info.experiment_id
art_dir = f"mlruns/{experiment_id}/{run_id}/artifacts"
ckpt_path = f"mlruns/{experiment_id}/{run_id}/last.ckpt" 
mlflow.log_params(vars(opt))
source_code = [i for i in os.listdir() if ".py" in i]
for i in source_code:
    shutil.copy(i, f"{art_dir}/{i}")

# Load from ckpt
if opt.resume_from_checkpoint is not None:
    checkpoint = torch.load(opt.resume_from_checkpoint)
    opt.epoch = checkpoint['current_epoch'] + 1
    netG_B2A.load_state_dict(checkpoint['netG_B2A'])
    netD_A.load_state_dict(checkpoint['netD_A'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
    print(f'find ckpt, load from checkpoint: {opt.resume_from_checkpoint}, epoch is {opt.epoch}')

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
#  Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size, opt.size).to(device)
input_B = torch.FloatTensor(opt.batch_size, opt.output_nc, opt.size, opt.size).to(device)
target_real = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.RandomResizedCrop(256, scale=(0.6, 1.4), interpolation=Image.BICUBIC),
               transforms.RandomHorizontalFlip(p=0.2),
               transforms.ColorJitter(0.2, 0.2),
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,)),
               ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

simple_dl = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=4, shuffle=False, num_workers=opt.n_cpu)
fix_sample = next(iter(simple_dl))
fix_A = fix_sample['A'].to(device)
fix_B = fix_sample['B'].to(device)

###################################

###### Training ######
step = 0
for epoch in range(opt.epoch, opt.n_epochs):


    step += 1
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Total loss
        loss_G = loss_identity_A + loss_GAN_B2A
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################
        pbar.set_description(f'Epoch:{epoch}')
        pbar.set_postfix_str(f'loss={loss_G:.4}, idt={loss_identity_A:.4}, G={loss_GAN_B2A:.4},  D={loss_D_A:.4}')
        if i % opt.log_step == 0:
            mlflow.log_metrics({'loss_G': loss_G.item(), 'loss_G_identity': loss_identity_A.item(), 'loss_G_GAN': loss_GAN_B2A.item(), 'loss_D': loss_D_A.item()}, step=step)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()

    # Image sample
    with torch.no_grad():
        fake_A = netG_B2A(fix_B)
        imgcat = torch.cat((fix_B,fake_A),dim=2)
        imgcat = torchvision.utils.make_grid(imgcat,normalize=True)
        torchvision.utils.save_image(imgcat, f'{art_dir}/img_{str(epoch).zfill(4)}.png')

    # Save models checkpoints
    states = {'netG_B2A': netG_B2A.state_dict(),
              'netD_A': netD_A.state_dict(),
              'optimizer_G': optimizer_G.state_dict(),
              'optimizer_D_A': optimizer_D_A.state_dict(),
              'current_epoch': epoch}
    torch.save(states, ckpt_path)


mlflow.end_run()
