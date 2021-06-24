import glob
import os
import random

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def tophat(img, fsize=20):
    img = np.array(img).astype(np.float32)
    filterSize = (fsize, fsize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    wth = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    bth = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    dst = (img + wth - bth).clip(0, 255).astype(np.uint8)
    return T.ToPILImage()(dst)

def random_rotate(img, p=1, angles=[0, 90, 180, 270]):
    if random.random() < p:
        angle = random.choice(angles)
        return T.functional_pil.rotate(img, angle)
    else:
        return img


def random_gamma(img, scale=(0.1, 2.2), p=1):
    if random.random() < p:
        gamma = random.uniform(scale[0], scale[1])
        return T.functional_pil.adjust_gamma(img, gamma)
    else:
        return img


class ImageDataset(Dataset):
    def __init__(self, root, size=256, unaligned=False, mode='train', transform=None):
        if mode == 'train':
            self.transform = T.Compose([
                T.Lambda(lambda img: tophat(img,50)), # TopHat增强
                T.RandomResizedCrop(size, scale=(0.6, 1), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x:random_rotate(x, p=1)),
                T.ColorJitter(0.3, 0.3),
                T.Lambda(lambda x:random_gamma(x, (0.2, 2), p=1)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = T.Compose([
                T.Lambda(lambda img: tophat(img,50)), # TopHat增强
                T.Resize(size),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        if transform:
            self.transform = transform

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
