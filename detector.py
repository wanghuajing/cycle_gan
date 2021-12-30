import os
from options.test_options import TestOptions
from models import create_model
import cv2
import torch
import numpy as np


def pad(img, data=256):
    oh, ow = img.shape
    h = int(np.ceil(oh / data) * data)
    w = int(np.ceil(ow / data) * data)
    if h == oh and w == ow:
        return img
    img = cv2.copyMakeBorder(img, 0, h - oh, 0, w - ow, cv2.BORDER_REFLECT)
    return img


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    # --dataroot
    # /media/zhao/HD1/data/mammo300/all/
    opt.name = 'pix2pix_256_1'
    # --results_dir
    # ./datasets/

    opt.model = 'pix2pix'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.netG = 'unet_256'
    opt.preprocess = 'none'
    opt.no_flip = True
    opt.num_threads = 0
    opt.batch_size = 1
    opt.gpu_ids = [1]
    opt.dataset_mode = 'add'
    opt.norm = 'batch'
    opt.epoch = 500
    opt.direction = 'AtoB'
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    img = cv2.imread('demo.png', -1)
    img256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img256 = torch.from_numpy((img256 / 65535.0).astype(np.float32))
    img256 = img256.unsqueeze(0)
    img256 = img256.unsqueeze(0)
    data = (img256 - 0.31534) / 0.08676
    # dataset = create_dataset(opt)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()
    # model.set_input(data)  # unpack data from data loader
    data = data.to(device)
    img256 = model.netG(data)  # run inference
    img256 = img256.cpu().detach().numpy()
    img256 = ((img256[0, 0, :, :] * 0.5 + 0.5) * 65535.0).astype(np.uint16)
    img_add = cv2.resize(img256, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # u_net配置
    opt.input_nc = 2
    opt.name = 'pix2pix_add_final'
    opt.epoch = 400

    img = pad(img, 256)
    img_add = pad(img_add, 256)

    img = torch.from_numpy((img / 65535.0).astype(np.float32))
    img_add = torch.from_numpy((img_add / 65535.0).astype(np.float32))
    img = img.unsqueeze(0)
    img_add = img_add.unsqueeze(0)
    data = torch.cat((img, img_add), 0)
    data = data.unsqueeze(0)
    data = (data - 0.5) / 0.5
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()
    # model.set_input(data)  # unpack data from data loader
    data = data.to(device)
    img_out = model.netG(data)  # run inference
    img_out = img_out.cpu().detach().numpy()
    img_out = ((img_out[0, 0, :, :] * 0.5 + 0.5) * 65535.0).astype(np.uint16)

    cv2.imwrite('3.png', img_out)
