import numpy as np
import pydicom
from glob import glob
import os
import cv2
import mammo.intensity as intensity
import mammo.segmentation as segmentation
from tqdm import tqdm
from skimage.measure import label
from tqdm import tqdm
import pandas as pd
import numbers
from numpy.lib.stride_tricks import as_strided
import pydicom
from PIL import Image

'''
一些工具函数
'''


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


def patchify(arr_in, window_shape, step=1):
    '''
    将图像裁片的代码
    :param arr_in:
    :param window_shape:
    :param step:
    :return:
    '''

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
                                (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
                        ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def run_make_full_img_dataset():
    '''
    将dicom的raw和proc转换为png,全图
    :return:
    '''
    OD_MAX = 3.2
    OD_MIN = 0.2
    root_dcm_dir = '/media/zhao/HD1/data/mammo300/'
    res_dir = '//media/zhao/HD1/data/ai-postprocess/mammo300_png/full'
    os.makedirs(os.path.join(res_dir, 'od'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'proc'), exist_ok=True)
    df = pd.read_csv('/media/zhao/HD1/data/mammo300/raw_dcm.csv')
    df1 = pd.read_csv('/media/zhao/HD1/data/mammo300/proc_dcm.csv')
    for i in tqdm(range(len(df))):
        raw_dcm_path = df['image_path'][i]
        basename = os.path.basename(raw_dcm_path)
        basename = rchop(basename, '_raw.dcm')
        proc_dcm_path = df1['image_path'][i]
        ds = pydicom.dcmread(root_dcm_dir + raw_dcm_path)
        img_raw = ds.pixel_array.astype(np.float32)
        ds = pydicom.dcmread(root_dcm_dir + proc_dcm_path)
        img_proc = ds.pixel_array.astype(np.float32)
        fg_mask = (img_proc > 0).astype(np.uint8)
        # 找到raw图背景均值
        bk_median = np.median(img_raw[fg_mask == 0])
        # 计算od
        img_od = np.log10(bk_median + 1) - np.log10(img_raw + 1)
        img_od[img_od < OD_MIN] = 0
        img_od[img_od > OD_MAX] = OD_MAX
        cv2.imwrite(os.path.join(res_dir, 'proc', basename + '.png'), ((img_proc / 4095.) * 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(res_dir, 'od', basename + '.png'), ((img_od / OD_MAX) * 65535).astype(np.uint16))


def gen_img_csv():
    return


def run_resize_data():
    size_dst = (256, 256)
    cut = 20  # 裁减上下边框
    root_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/full'
    res_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/%dx%d' % (size_dst[0], size_dst[1]) + '/cut%d' % (cut)
    os.makedirs(os.path.join(res_dir, 'od'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'proc'), exist_ok=True)
    od_paths = glob(os.path.join(root_dir, 'od', '*.png'))
    for od_path in tqdm(od_paths):
        basename = os.path.basename(od_path)
        im = cv2.imread(od_path, -1)
        if cut > 0:
            im = im[cut:-cut, :]
        im = cv2.resize(im, size_dst, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(res_dir, 'od', basename), im)
    proc_paths = glob(os.path.join(root_dir, 'proc', '*.png'))
    for proc_path in tqdm(proc_paths):
        basename = os.path.basename(proc_path)
        im = cv2.imread(proc_path, -1)
        if cut > 0:
            im = im[cut:-cut, :]
        im = cv2.resize(im, size_dst, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(res_dir, 'proc', basename), im)
    return


def run_make_alinged_csv_dataset():
    '''
    制作alinged_dataset的csv文件
    :return:
    '''
    root_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/cut20/'
    root_dir_od = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/cut20/od'
    root_dir_proc = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/cut20/proc'
    out_csv_path = os.path.join(root_dir, 'aligned_full.csv')
    od_paths = glob(os.path.join(root_dir_od, '*.png'))
    od_paths.sort()
    proc_paths = glob(os.path.join(root_dir_proc, '*.png'))
    proc_paths.sort()
    od_paths = [x[len(root_dir):] for x in od_paths]
    proc_paths = [x[len(root_dir):] for x in proc_paths]
    df = pd.DataFrame({'od': od_paths, 'proc': proc_paths})
    df.to_csv(out_csv_path, index=False)
    a = 1
    return


def run_get_stats():
    '''
    统计od和proc图的前景区域均值和方差
    :return:
    '''
    root_dir_od = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/cut20/od'
    od_paths = glob(os.path.join(root_dir_od, '*.png'))
    pix_intensities = np.array((1, 0))
    for idx, od_path in enumerate(od_paths):
        print(idx)
        im = cv2.imread(od_path, -1)
        if idx == 0:
            pix_array = im[im > 0]
        else:
            pix_array = np.concatenate((pix_array, im[im > 0]))
    mu = np.mean(pix_array)
    std = np.std(pix_array)
    print(mu)
    print(std)
    print("mu=%d, std=%d" % (mu, std))
    # mu=20666, std=5686
    # mu = 0.31534294651712824
    # std = 0.08676279850461585

    return


def make_pix2pix_dataset():
    '''
    构造pix2pix中的aligned Dataset
    :return:
    '''
    od_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/od'
    proc_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/proc'
    res_dir = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/256x256/split'
    os.makedirs(os.path.join(res_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'test'), exist_ok=True)
    od_paths = glob(os.path.join(od_dir, '*.png'))
    od_paths.sort()
    for idx, od_path in enumerate(od_paths):
        print(idx)
        basename = os.path.basename(od_path)
        proc_path = os.path.join(proc_dir, basename)
        im_A = cv2.imread(od_path, -1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_B = cv2.imread(proc_path, -1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_AB = np.concatenate([im_B, im_A], 1)
        if idx <= 1000:
            cv2.imwrite(os.path.join(res_dir, 'train', basename), im_AB)
        else:
            cv2.imwrite(os.path.join(res_dir, 'val', basename), im_AB)
    return


def make_cropped_dataset():
    '''
    制作裁片数据集,
    :return:
    '''
    P_SIZE = (512, 512)  # 裁片大小
    OVERLAP = 256  # 裁片位移量
    proc_dir = '/media/zhao/HD1/data/mammo300/all/'
    raw_dir = '/media/zhao/HD1/data/mammo300/all/'
    add_dir = '/media/zhao/HD1/data/mammo300/all/'
    res_root_dir = '/media/zhao/HD1/data/mammo300/all/crop/'
    res_root_dir = os.path.join(res_root_dir, 'patch_%dx%d_o%d') % (P_SIZE[0], P_SIZE[1], OVERLAP)
    os.makedirs(res_root_dir, exist_ok=True)
    os.makedirs(os.path.join(res_root_dir, 'proc'), exist_ok=True)
    os.makedirs(os.path.join(res_root_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(res_root_dir, 'add'), exist_ok=True)

    df = pd.read_csv('/media/zhao/HD1/data/mammo300/all/train.csv')
    for index, proc_path in tqdm(df.iterrows()):
        proc_path = proc_dir + proc_path['proc']
        basename = os.path.basename(proc_path)
        basename = basename.rstrip('.png')

        im_proc = cv2.imread(proc_path, -1)
        od_path = os.path.join(raw_dir + 'raw', basename + '.png')
        im_od = cv2.imread(od_path, -1)
        add_path = os.path.join(add_dir + 'add', basename + '.png')
        im_add = cv2.imread(add_path, -1)
        im_c = np.stack((im_proc, im_od, im_add), axis=2)
        im_c = im_c[20:-20, :, :]

        patchs = patchify(im_c, (P_SIZE[0], P_SIZE[1], 3), OVERLAP)
        patchs = np.reshape(patchs, (np.prod(patchs.shape[0:3]), patchs.shape[3], patchs.shape[4], patchs.shape[5]))
        for idx in range(patchs.shape[0]):
            im_patch = patchs[idx, :, :, :]
            # 去除背景占比小于0.4的图
            if np.mean((im_patch[:, :, 1] > 0).astype(np.float32)) < 0.3:
                continue
            cv2.imwrite(os.path.join(res_root_dir, 'proc', basename + '_%03d' % (idx) + '.png'),
                        im_patch[:, :, 0])
            cv2.imwrite(os.path.join(res_root_dir, 'raw', basename + '_%03d' % (idx) + '.png'),
                        im_patch[:, :, 1])
            cv2.imwrite(os.path.join(res_root_dir, 'add', basename + '_%03d' % (idx) + '.png'),
                        im_patch[:, :, 2])
    return


def run_make_xpect_ld():
    '''
    根据xpectvision自家机器出图,计算光密度
    :return:
    '''
    root_dir = '/media/zhao/HD1/data/ai-postprocess/20211117_sz'
    res_dir = os.path.join(root_dir, 'result')
    os.makedirs(res_dir, exist_ok=True)
    raw_paths = glob(os.path.join(root_dir, '*', '*_init.dcm'))
    for idx, raw_path in enumerate(raw_paths):
        print(idx)
        ds = pydicom.dcmread(raw_path)
        img = ds.pixel_array.astype(np.float32)
        # 进行sqrt反变换
        img = (img / 128) ** 2
        # 分割前景背景
        i_low, i_high = np.percentile(img, (1, 99))
        img_n = (img - i_low) / (i_high - i_low)
        img_n[img_n > 1] = 1
        img_n[img_n < 0] = 0
        img_n = 1 - img_n
        img_n = (img_n * 65535).astype(np.uint16)
        ret, th = cv2.threshold(img_n, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bk_mask = img_n < ret
        # 计算od
        bk_count = np.median(img[bk_mask > 0])
        img_od = np.log10(bk_count + 1) - np.log10(img + 1)
        OD_MAX = 3.2
        OD_MIN = 0.2
        img_od[img_od > OD_MAX] = OD_MAX
        img_od[img_od < OD_MIN] = 0
        cv2.imwrite(os.path.join(res_dir, '%02d.png' % (idx)), ((img_od / OD_MAX) * 65536).astype(np.uint16))

    return


def low2high():
    '''
    将低分辨率的图片转成原始分辨率
    '''

    path = '/media/zhao/HD1/data/mammo300/all/'
    df = pd.read_csv(path + 'test.csv')
    for index, add in tqdm(df.iterrows()):
        img = Image.open(path + add['add']).convert('I')
        img = img.resize((3328, 4096), 3)
        img.save(path + add['add'])


if __name__ == '__main__':
    # 将dicom的raw和proc转换为png,全图
    # run_make_full_img_dataset()

    # 将上面数据集resize为256*256，用于全局训练
    # run_resize_data()

    # 制作csv版本的aligned_dataset
    # run_make_alinged_csv_dataset()

    # 计算od图的均值和方差
    # run_get_stats()

    # 变成pix2pix默认的aligned_dataset格式
    # make_pix2pix_dataset()

    # 制作裁片数据集
    # make_cropped_dataset()

    # 河南数据转换为第分辨率od
    # run_make_xpect_ld()
    path = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/demo/'
    df = pd.read_csv(path + 'test.csv')
    for index, item in df.iterrows():
        od = cv2.imread(path + item['od'], -1)
        new = cv2.imread(path + item['new'], -1)
        new = cv2.resize(new, (od.shape[1], od.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/media/zhao/HD1/data/ai-postprocess/mammo300_png/demo2/' + item['new'], new)
