import numpy as np
import pydicom
from glob import glob
import os
import cv2
import mammo.intensity as intensity
import mammo.segmentation as segmentation
from tqdm import tqdm
from skimage.measure import label

W_PARAMS = {'p0':0.3, 'p1':0.98, 'y0':0.25,'y1':0.62} # 将p分位的像素亮度映射至y
WW_FACTOR = 1 # 这个参数用于调节柔和度。例如如果医生认为对比度过强，则设置这个参数小于1. 过弱则大于1.
BITS = 16
S_WIN = True # 是否使用sigmoid window。如果为false，直接将映射后的像素值写入dicom pixel array

def solve_windows(x0,y0,x1,y1):
    '''
    sigmoid window定义为：y = 1/(1+np.exp(-4*(x-wc)/ww))
    给定过该sigmoid曲线的两个点[x0,y0],[x1,y1],求解wc,ww参数
    :param x0:
    :param y0:
    :param x1:
    :param y1:
    :return:
    '''
    a0 = 0.25*np.log(y0/(1-y0))
    a1 = 0.25*np.log(y1/(1-y1))
    b0 = b1 = 1
    c0 = x0
    c1 = x1
    # A = np.array([[a0,a1],[b0,b1]])
    A = np.array([[a0,b0],[a1,b1]])
    D = np.array([c0,c1]).T
    res = np.linalg.solve(A,D)
    ww, wc = res[0], res[1]
    return ww, wc

def prep_ge_full():
    src_dir = r'F:\Henan_Mammo\GE_annotated\2014'  # 这里使用2014的图片，共有4000多张，2017年开始，探测器有一条坏线
    res_dir = r'F:\Data\Mammo\style\GE_full'
    os.makedirs(res_dir, exist_ok=True)
    dcm_files = glob(os.path.join(src_dir, '*/*/*.dcm'))
    dcm_files = dcm_files[:1200]
    for dcm_file in tqdm(dcm_files):
        basename = os.path.basename(dcm_file)
        ds = pydicom.dcmread(dcm_file)
        img = ds.pixel_array.astype(np.float32)
        # 分割乳房区域的函数。GE的图像有个别的背景不为0，因此需要阈值分割
        fgmask, bmask = segmentation.segbreast_prewitt(img / img.max())
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # 将mask膨胀操作，留出一些余量。
        bmask_dilate = cv2.morphologyEx(bmask, cv2.MORPH_DILATE, kernel, iterations=10)
        # 找出mask 的bounding box, 并且裁切保存
        nz_x = np.where(np.sum(bmask_dilate, axis=0))
        nz_y = np.where(np.sum(bmask_dilate, axis=1))
        x_min, x_max = nz_x[0].min(), nz_x[0].max()
        y_min, y_max = nz_y[0].min(), nz_y[0].max()
        img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
        cv2.imwrite(os.path.join(res_dir, basename + '.png'), ((img_crop / 4095.) * 65535).astype(np.uint16))
    return

def prep_ge_full_sigmoid():
    src_dir = r'F:\Henan_Mammo\GE_annotated\2014'  # 这里使用2014的图片，共有4000多张，2017年开始，探测器有一条坏线
    res_dir = r'F:\Data\Mammo\style\GE_full_sigmoid'
    os.makedirs(res_dir, exist_ok=True)
    dcm_files = glob(os.path.join(src_dir, '*/*/*.dcm'))
    dcm_files = dcm_files[:1200]
    for dcm_file in tqdm(dcm_files):
        basename = os.path.basename(dcm_file)
        ds = pydicom.dcmread(dcm_file)
        img = ds.pixel_array.astype(np.float32)
        # 分割乳房区域的函数。GE的图像有个别的背景不为0，因此需要阈值分割
        fgmask, bmask = segmentation.segbreast_prewitt(img / img.max())
        # 对于MLO位做一些处理
        view = ds.ViewPosition
        if view == 'MLO':
            cmask = segmentation.get_cmask(bmask)
        else:
            cmask = bmask
        # sigmoid窗
        p0, p1 = W_PARAMS['p0'] * 100, W_PARAMS['p1'] * 100
        y0, y1 = W_PARAMS['y0'], W_PARAMS['y1']
        x0, x1 = np.percentile(img[cmask>0], (p0, p1))
        ww, wc = intensity.solve_windows(x0, y0, x1, y1)
        ww = ww * WW_FACTOR  # 根据医生喜好调节窗宽一点还是窄一点。
        lut = intensity.make_sigmoid_lut(wc, ww, input_range=2 ** 12, out_max=2 ** BITS - 1)
        img = lut[img.astype(np.uint16)]
        # 将mask膨胀操作，留出一些余量。
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        bmask_dilate = cv2.morphologyEx(bmask, cv2.MORPH_DILATE, kernel, iterations=10)
        # 找出mask 的bounding box, 并且裁切保存
        nz_x = np.where(np.sum(bmask_dilate, axis=0))
        nz_y = np.where(np.sum(bmask_dilate, axis=1))
        x_min, x_max = nz_x[0].min(), nz_x[0].max()
        y_min, y_max = nz_y[0].min(), nz_y[0].max()
        img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
        cv2.imwrite(os.path.join(res_dir, basename + '.png'), (img_crop).astype(np.uint16))
    return

def prep_hlg_full():
    src_dir = r'F:\Data\Mammo\mammo300\dcm'  # 这里使用2014的图片，共有4000多张，2017年开始，探测器有一条坏线
    res_dir = r'F:\Data\Mammo\style\HLG_full'
    os.makedirs(res_dir, exist_ok=True)
    dcm_files = glob(os.path.join(src_dir, '*/*_proc.dcm'))
    dcm_files = dcm_files[:1200]
    for dcm_file in tqdm(dcm_files):
        basename = os.path.basename(dcm_file)
        ds = pydicom.dcmread(dcm_file)
        img = ds.pixel_array.astype(np.float32)
        img = img[20:-20, :]
        # 分割乳房区域的函数。GE的图像有个别的背景不为0，因此需要阈值分割
        fgmask = (img > 0).astype(np.uint8)
        labels = label(fgmask)
        assert (labels.max() != 0)  # assume at least 1 CC
        bmask = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        bmask = bmask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # 将mask膨胀操作，留出一些余量。
        bmask_dilate = cv2.morphologyEx(bmask, cv2.MORPH_DILATE, kernel, iterations=10)
        # 找出mask 的bounding box, 并且裁切保存
        nz_x = np.where(np.sum(bmask_dilate, axis=0))
        nz_y = np.where(np.sum(bmask_dilate, axis=1))
        x_min, x_max = nz_x[0].min(), nz_x[0].max()
        y_min, y_max = nz_y[0].min(), nz_y[0].max()
        img_crop = img[y_min:y_max + 1, x_min:x_max + 1]
        scale_f = 0.7
        h, w = img_crop.shape
        h_new, w_new = int(h*0.7), int(w*0.7)
        img_crop = cv2.resize(img_crop, (w_new, h_new))
        cv2.imwrite(os.path.join(res_dir, basename + '.png'), ((img_crop / 4095.) * 65535).astype(np.uint16))
    return

if __name__ == '__main__':
    # prep_ge_full()
    # prep_hlg_full()
    prep_ge_full_sigmoid()

