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
from models import UNet
from utils import LambdaLR, ReplayBuffer, weights_init_normal, EMA

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/cycledsa_v2/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
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
#  netG_A2B = Generator(opt.input_nc, opt.output_nc)
#  netG_B2A = Generator(opt.output_nc, opt.input_nc)
netG_A2B = UNet(opt.input_nc, opt.output_nc, dim=32)
netG_B2A = UNet(opt.output_nc, opt.input_nc, dim=32)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# EMA model
ema_updater = EMA(0.995)
netE = UNet(opt.output_nc, opt.input_nc, dim=32)

if opt.device != 'cpu':
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)
    netE.to(device)

#  netG_A2B.apply(weights_init_normal)
#  netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Set logger
mlflow.set_experiment(opt.exp_name)
run = mlflow.start_run(run_name=opt.name)
run_id = run.info.run_id
print('run id: ',run_id)
experiment_id = run.info.experiment_id
run_dir = f'mlruns/{experiment_id}/{run_id}'
art_dir = f"{run_dir}/artifacts"
ckpt_path = f"{run_dir}/last.ckpt" 
mlflow.log_params(vars(opt))
source_code = [i for i in os.listdir() if ".py" in i]
for i in source_code:
    shutil.copy(i, f"{art_dir}/{i}")

# Load from ckpt
if opt.resume_from_checkpoint is not None:
    checkpoint = torch.load(opt.resume_from_checkpoint)
    opt.epoch = checkpoint['current_epoch'] + 1
    netG_A2B.load_state_dict(checkpoint['netG_A2B'])
    netG_B2A.load_state_dict(checkpoint['netG_B2A'])
    netD_A.load_state_dict(checkpoint['netD_A'])
    netD_B.load_state_dict(checkpoint['netD_B'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
    print(f'find ckpt, load from checkpoint: {opt.resume_from_checkpoint}, epoch is {opt.epoch}')

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
#  Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size, opt.size).to(device)
input_B = torch.FloatTensor(opt.batch_size, opt.output_nc, opt.size, opt.size).to(device)
target_real = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.RandomResizedCrop(opt.size, scale=(0.6, 1.4), interpolation=Image.BICUBIC),
               transforms.RandomHorizontalFlip(p=0.2),
               transforms.ColorJitter(0.2, 0.2),
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,drop_last=True)

simple_dl = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=8, shuffle=False, num_workers=opt.n_cpu)
fix_sample = next(iter(simple_dl))
fix_A = fix_sample['A'].to(device)
fix_B = fix_sample['B'].to(device)

###################################

###### Training ######
step = 0
for epoch in range(opt.epoch, opt.n_epochs):
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        step += 1
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        #  G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
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

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # ema update
        if step!=0 and step % 10 == 0 and step > 200:
            ema_updater.update_moving_average(netE, netG_B2A)

        pbar.set_description(f'Epoch:{epoch}')
        pbar.set_postfix_str(f'loss={loss_G:.4}, idt={loss_identity_A + loss_identity_B:.4}, G={loss_GAN_A2B + loss_GAN_B2A:.4}, cycle={loss_cycle_ABA + loss_cycle_BAB:.4}, D={loss_D_A + loss_D_B:.4}')
        if i % opt.log_step == 0:
            mlflow.log_metrics({'loss_G': loss_G.item(), 'loss_G_identity': (loss_identity_A + loss_identity_B).item(), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(), 'loss_D': (loss_D_A + loss_D_B).item()}, step=step)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Image sample
    with torch.no_grad():
        #  fake_B = netG_A2B(fix_A)
        out = netG_B2A.model(fix_B)
        fake_A = out + fix_B
        out2 = (out>0.01).float()
        
        out_ema = netE.model(fix_B)
        fake_E = out_ema + fix_B
        out_ema2 = (out_ema>0.01).float()
        
        #  imgs = torch.cat((fix_A,fake_B,fix_B,fake_A),dim=2)
        imgs = torch.cat((fix_B, fake_A, out, out2, out_ema, out_ema2),dim=2)
        imgs = torchvision.utils.make_grid(imgs,normalize=True,nrow=8)
        torchvision.utils.save_image(imgs, f'{art_dir}/img_{str(epoch).zfill(4)}.png')

    # Save last checkpoints
    states = {'netG_A2B': netG_A2B.state_dict(),
              'netG_B2A': netG_B2A.state_dict(),
              'netD_A': netD_A.state_dict(),
              'netD_B': netD_B.state_dict(),
              'netE': netE.state_dict(),
              'optimizer_G': optimizer_G.state_dict(),
              'optimizer_D_A': optimizer_D_A.state_dict(),
              'optimizer_D_B': optimizer_D_B.state_dict(),
              'current_epoch': epoch}
    torch.save(states, ckpt_path)

    # save every epoch for B2A
    states = {'netG_B2A': netG_B2A.state_dict(),
              'netE': netE.state_dict()}
    torch.save(states, f'{run_dir}/{str(epoch).zfill(3)}.ckpt')

mlflow.end_run()
