''' Basic packages
'''
import os
import cv2
import glob
import imageio
import numpy as np
from Data_zoo import common_utils as cutils
''' PyTorch package
'''
import torch
import torch.utils.data as Tdata



''' DATASET class
'''

class create_dataset(Tdata.Dataset):
    def __init__(self, args, phase=None):
        super(create_dataset, self).__init__()
        self.dir_root = args.dir_root
        self.dir_hr = args.dir_hr
        self.dir_lr = args.dir_lr
        self.phase = args.phase if phase==None else phase
        self.scale = args.scale
        self.augmentation = args.do_augmentation
        self.n_colors = args.n_colors

        self.images_hr, self.images_lr = self._scan()
        self.repeat = args.test_every//(len(self.images_hr)//args.batch_size) if self.phase=='train' else 1

    ''' intrinsic function
    '''
    def _scan(self, phase='train'):
        if phase=='train':
            hr_list = sorted(glob.glob(os.path.join(self.dir_root, self.dir_hr, '*.png')))
            lr_list = sorted(glob.glob(os.path.join(self.dir_root, self.dir_lr, '*.png')))
            return hr_list, lr_list
        else:
            hr_list = sorted(glob.glob(os.path.join(self.dir_root, self.dir_hr_eval, '*.png')))
            lr_list = sorted(glob.glob(os.path.join(self.dir_root, self.dir_lr_eval, '*.png')))
            return hr_list, lr_list
    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename = f_lr.split('/')[-1]  #0001x4.png
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)
        return hr, lr, filename
    def _get_patch(self, hr, lr):
        if self.phase == 'train':
            lr, hr = cutils.get_patch(lr, hr, scale=self.scale, multi_scale=False)
            if self.augmentation:
                lr, hr = cutils.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih*self.scale, 0:iw*self.scale]
        return hr, lr
    
    def __len__(self):
        return len(self.images_hr) * self.repeat

    def __getitem__(self, idx):
        hr, lr, filename = self._load_file(idx)
        hr, lr = self._get_patch(hr, lr)
        hr, lr = cutils.set_channel(hr, lr, n_channels=self.n_colors)
        hr_tensor, lr_tensor = cutils.np2Tensor(hr, lr, rgb_range=255)

        return hr_tensor, lr_tensor, filename



