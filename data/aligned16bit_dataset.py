import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import pandas as pd


class Aligned16bitDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.add_path = opt.dataroot
        self.paths = pd.read_csv(self.add_path + opt.phase + '.csv')
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.add_path + self.paths['raw'][index]
        A = Image.open(A_path).convert('I')
        B_path = self.add_path + self.paths['proc'][index]
        B = Image.open(B_path).convert('I')
        # split AB image into A and B

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        # 随机乘 [0.8 1.2]
        A = torch.from_numpy((np.array(A) / 65535.0).astype(np.float32))
        B = torch.from_numpy((np.array(B) / 65535.0).astype(np.float32))
        # A = A.unsqueeze(0)
        # B = B.unsqueeze(0)
        # 将A,B数据集分别标准化
        # A = (A - 0.402942) / 0.130789
        # B = (B - 0.304032) / 0.182379
        ##  for OD dataset   # mu = 0.31534294651712824 std = 0.08676279850461585
        A = (A - 0.5) / 0.5
        # A = (A - 0.31534) / 0.08676
        B = (B - 0.5) / 0.5

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)