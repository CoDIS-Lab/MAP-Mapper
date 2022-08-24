# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: dataloader.py includes the appropriate data loader for
             multi-label classification.
'''

import os

import rasterio
import torch
import random
import numpy as np
from tqdm import tqdm
from os.path import dirname as up
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from utils.paths import base_path
from patchify import patchify
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Pixel-level number of negative/number of positive per class
pos_weight = torch.Tensor([2.65263158, 27.91666667, 11.39285714, 18.82857143, 6.79775281,
                           6.46236559, 0.60648148, 27.91666667, 22.13333333, 5.03478261,
                           17.26315789, 29.17391304, 16.79487179, 12.88, 9.05797101])

bands_mean = np.array(np.array([0.03163572, 0.03457443, 0.03436435, 0.02358126]).astype('float32'))

bands_std = np.array([0.04967381, 0.06458357, 0.07120246, 0.05111466]).astype('float32')

###############################################################
# Multi-label classification Data Loader                      #
###############################################################
dataset_path = os.path.join(up(up(up(__file__))), 'data')


class GenDEBRIS_ML(Dataset):  # Extend PyTorch's Dataset class
    def __init__(self, mode='train', transform=None, standardization=None, path=dataset_path, agg_to_water=True):

        if mode == 'train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'), dtype='str')

        elif mode == 'test':
            self.ROIs = np.genfromtxt(os.listdir("/data/patches"), dtype='str')

        elif mode == 'val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'), dtype='str')

        else:
            raise

        self.X = []  # Loaded Images
        self.meta = []

        for roi in tqdm(self.ROIs, desc='Load ' + mode + ' set to memory'):
            roi_file = os.path.join(base_path, 'data', "patches", roi)  # Get File path

            # Load Image
            ds = rasterio.open(roi_file)
            self.meta.append(ds.meta)
            temp = np.copy(ds.read())
            ds = None
            size = 32
            mask_patches = patchify(temp, (size, size), step=size)

            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    patch = mask_patches[i, j, :, :]
                    self.X.append(patch)


        self.impute_nan = np.tile(bands_mean, (temp.shape[1], temp.shape[2], 1))
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.X)
        self.path = path

    def __len__(self):

        return self.length

    def getnames(self):
        return self.ROIs

    def __getitem__(self, index):

        img = self.X[index]

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')  # CxWxH to WxHxC

        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]

        if self.transform is not None:
            img = img.astype('float32')
            img = self.transform(img)

        if self.standardization is not None:
            img = self.standardization(img)

        return img


###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


###############################################################
# Weighting Function                                          #
###############################################################
def gen_weights(pos_weight, c=1.4):
    return 1 / torch.log(c + pos_weight)