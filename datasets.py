import glob
import os
import random
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensor
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# tv
class ImageDataset(Dataset):
    def __init__(self, root, size=256, unaligned=False, mode='train'):
        if mode == 'train':
            self.transform = T.Compose([
                T.RandomResizedCrop(size, scale=(0.6, 1.4), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.2),
                T.ColorJitter(0.2, 0.2),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))])
        else:
            self.transform = T.Compose([
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))])
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


# album
#  class ImageDataset(Dataset):
    #  def __init__(self, root, size=256, unaligned=False, mode='train', transforms_=None,):
        #  self.unaligned = unaligned

        #  self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        #  self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        #  if mode == 'train':
            #  self.transform = A.Compose([
                #  A.PadIfNeeded(int(size * 1.2), int(size * 1.2),border_mode=0),
                #  A.RandomResizedCrop(size, size, scale=(0.6, 1), p=0.4),
                #  A.Resize(size, size),
                #  A.HorizontalFlip(p=0.2),
                #  A.ColorJitter(p=0.3),
                #  A.RandomGamma(gamma_limit=(50, 150), p=0.2),
                #  A.Normalize((0.5,), (0.5,)),
                #  ToTensor()
            #  ])
        #  else:
            #  self.transform = A.Compose([
                #  A.Resize(size, size),
                #  A.Normalize((0.5,), (0.5,)),
                #  ToTensor()
            #  ])


    #  def __getitem__(self, index):
        #  item_A = np.array(Image.open(self.files_A[index % len(self.files_A)]).convert('L'))
        #  item_A = self.transform(image=item_A)['image']

        #  if self.unaligned:
            #  item_B = np.array(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
            #  item_B = self.transform(image=item_B)['image']
        #  else:
            #  item_B = np.array(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))
            #  item_B = self.transform(image=item_B)['image']

        #  return {'A': item_A.T.unsqueeze(0), 'B': item_B.T.unsqueeze(0)}

    #  def __len__(self):
        #  return min(len(self.files_A), len(self.files_B))
