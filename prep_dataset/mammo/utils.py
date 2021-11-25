import matplotlib.pyplot as plt
import os
from scipy import ndimage
import numpy as np
import cv2
import rawpy
''' 一些工具函数 '''
def save_raw(filename, img, format='float32',dir='tmp'):
    os.makedirs(dir, exist_ok=True)
    if format == 'float32':
        img.astype(np.float32).tofile(os.path.join(dir, filename) + str(img.shape) + '.raw')
    elif format == 'uint16':
        img.astype(np.uint16).tofile(os.path.join(dir, filename) + str(img.shape) + '.raw')
    elif format == 'uint8':
        img.astype(np.uint8).tofile(os.path.join(dir, filename) + str(img.shape) + '.raw')
    else:
        raise Exception("format not supported!")
    return

def rot_dcm(ds):
    '''
    旋转dicom图像，把图像从底边转到正确的侧边位置。
    :param ds:
    :return:
    '''
    img_raw = ds.pixel_array
    # if ds['ImageLaterality'].value == 'R':
    #     img_raw = np.rot90(img_raw)
    # else:
    #     img_raw = np.rot90(img_raw, 3)
    if ds['ProtocolName'].value[0] == 'R':
        img_raw = np.rot90(img_raw)
    else:
        img_raw = np.rot90(img_raw, 3)
    ds.PixelData = img_raw.tobytes()
    ds[0x280010].value = img_raw.shape[0] # rows
    ds[0x280011].value = img_raw.shape[1] # columns
    return ds

def write_dcm(path_dcm, ds, img_float,border=20, bits=12, w_min=0, w_max=1):
    '''
    写入dicom的函数
    :param path_dcm:
    :param ds:
    :param img_float:
    :param border:
    :param bits:
    :return:
    '''
    # 读取hologic图片时把上下20像素的白条去掉了，补回来
    img_float = np.concatenate((np.zeros((border, img_float.shape[1])).astype(float),
                                       img_float,
                                       np.zeros((border, img_float.shape[1])).astype(float)), axis=0)
    ds.pixel_array[:, :] = (img_float * (2 ** bits - 1)).astype(np.uint16)
    ds.PixelData = ds.pixel_array.tobytes()
    # 设置其他dicom tags
    wc = int((w_min + w_max)/2 * 2 ** bits)
    ww = int((w_max - w_min) * 2 ** bits)

    ds[0x280100].value = 16  # bit stored
    ds[0x280101].value = bits  # bit allocated
    ds[0x280102].value = bits - 1  # high bit
    ds[0x281041].value = -1  # PixelIntnesityRelationshipSign
    ds[0x281040].value = 'LIN'
    ds[0x281050].value = wc  # WindowCenter
    ds[0x281051].value = ww  # WindowWidth
    ds[0x20500020].value = 'IDENTITY'  # 'PresentationLUTShape'
    ds['PresentationIntentType'].value = 'FOR PRESENTATION'
    ds[0x00280004].value = 'MONOCHROME2'
    ds.save_as(path_dcm)
    return

def write_dcm_sigmoid(path_dcm, ds, img_float,border=20, bits=12, wc=1500,ww=650):
    '''
    写入dicom的函数
    :param path_dcm:
    :param ds:
    :param img_float:
    :param border:
    :param bits:
    :return:
    '''
    # 读取hologic图片时把上下20像素的白条去掉了，补回来
    img_float = np.concatenate((np.zeros((border, img_float.shape[1])).astype(float),
                                       img_float,
                                       np.zeros((border, img_float.shape[1])).astype(float)), axis=0)
    ds.pixel_array[:, :] = (img_float * (2 ** bits - 1)).astype(np.uint16)
    ds.PixelData = ds.pixel_array.tobytes()
    ww = ww*(2**bits - 1)
    if ww < 1:
        ww = 1
    wc = wc*(2**bits - 1)
    if wc < 1:
        wc = 1
    # 设置其他dicom tags
    ds.VOILUTFunction = 'SIGMOID'
    ds[0x280100].value = 16  # bit stored
    ds[0x280101].value = bits  # bit allocated
    ds[0x280102].value = bits - 1  # high bit
    ds[0x281041].value = -1  # PixelIntnesityRelationshipSign
    ds[0x281040].value = 'LIN'
    ds[0x281050].value = wc  # WindowCenter
    ds[0x281051].value = ww  # WindowWidth
    ds[0x20500020].value = 'IDENTITY'  # 'PresentationLUTShape'
    ds['PresentationIntentType'].value = 'FOR PRESENTATION'
    ds[0x00280004].value = 'MONOCHROME2'
    ds.save_as(path_dcm)
    return

def write_dcm_sigmoid_1(path_dcm, ds, img_int, border=20, bits=12, wc=1500, ww=650, s_win=False):
    '''
    写入dicom的函数, 次函数
    :param path_dcm:
    :param ds:
    :param img_int:
    :param border:
    :param bits:
    :return:
    '''
    # 读取hologic图片时把上下20像素的白条去掉了，补回来
    img_int = np.concatenate((np.zeros((border, img_int.shape[1])).astype(np.uint16),
                              img_int,
                              np.zeros((border, img_int.shape[1])).astype(np.uint16)), axis=0)
    ds.pixel_array[:, :] = img_int
    ds.PixelData = ds.pixel_array.tobytes()
    if ww < 1:
        ww = 1
    if wc < 1:
        wc = 1
    # 设置其他dicom tags
    if s_win:
        ds.VOILUTFunction = 'SIGMOID'
    ds[0x280100].value = 16  # bit stored
    ds[0x280101].value = bits  # bit allocated
    ds[0x280102].value = bits - 1  # high bit
    ds[0x281041].value = -1  # PixelIntnesityRelationshipSign
    ds[0x281040].value = 'LOG'
    ds[0x281050].value = wc  # WindowCenter
    ds[0x281051].value = ww  # WindowWidth
    ds[0x20500020].value = 'IDENTITY'  # 'PresentationLUTShape'
    ds['PresentationIntentType'].value = 'FOR PRESENTATION'
    ds[0x00280004].value = 'MONOCHROME2'
    ds.save_as(path_dcm)
    return

class ImgProc():
    '''
    这个没必要实现，针对hologic去坏点的。
    '''
    def __init__(self):
        print('this is a group of image processing functions.')
        return
    @staticmethod
    def find_bp(img):

        '''
        remove bad point from raw image
        :param img:
        :return:
        '''
        img_median = cv2.medianBlur(img, 5)
        # local_var = generic_filter(img, np.std, size=5)
        img_diff = abs(img - img_median)
        bp = np.where(img_diff>0.1)
        return bp
    @staticmethod
    def remove_bp(img, latterality):
        # 目前检测出的hologic探测器的坏点
        bps = []
        for y in range(668,692):
            for x in range(1622, 1625):
                bps.append([y, x])
        bps = np.array(bps)

        if latterality == 'R':
            # 如果是右视图，则坏点是对称的
            bps = np.array([4096, 3328]) - bps - 1
        img_new = img
        for bp in bps:
            img_new[bp[0], bp[1]] = np.median(img[bp[0]-7:bp[0]+7, bp[1]-7:bp[1]+7])
        return img_new

def load_raw_ak(filepath, h=3072, w=4096, border=120, laterality='L'):
    raw = np.fromfile(filepath, dtype=np.uint16)
    raw = np.reshape(raw, (3072, 4096))
    raw = raw[:,border:-border]

    if laterality == 'R':
        img_raw = np.rot90(raw)
    else:
        img_raw = np.rot90(raw, 3)
    return img_raw

def load_raw(filepath, h=3072, w=4096, border=0):
    raw = np.fromfile(filepath, dtype=np.uint16)
    raw = np.reshape(raw, (3072, 4096))
    if border:
        raw = raw[:,border:-border]
    return raw


def load_bk_params(BG_table_path, BH_factor_path, kvp, mas):
    BG_table = np.genfromtxt(BG_table_path, delimiter=',')
    BH_factor = np.genfromtxt(BH_factor_path,delimiter=',')
    bg = mas*BG_table[BG_table[:,0]==kvp, 1][0]
    p = np.zeros(4)
    p[0:3] = BH_factor[BH_factor[:,0]==kvp, 1:].squeeze()
    return bg, p

def percentile_approx(img, q, roi=None, scale=0.15):
    '''
    快速计算图像roi中Percentile的方法。对大图像素亮度值进行排序时间复杂度高。因此先resize，再计算percentile。
    :param img: 输入图像
    :param q: 等同于np.percentile()函数中的q，是单一的数或者一个tuple,取值在0-100之间。例如,5,或者(5,95)
    :param roi: roi mask。uint8格式，roi内部取值为1，背景为0。
    :param scale: 缩放系数。
    :return: 等同于np.percentile的return.
    '''
    h, w = img.shape
    if roi is None:
        roi = np.ones((h, w), dtype=np.uint8)
    h_s, w_s = int(h*scale), int(w*scale)
    img_s = cv2.resize(img, (w_s,h_s))
    mask_s = cv2.resize(roi, (w_s, h_s))
    return np.percentile(img_s[mask_s>0], q)

def median_approx(img, roi, scale=0.1):
    '''
    近似取图像roi区域内的中位数。对大图像素亮度值进行排序时间复杂度高。因此先resize，再计算中位数。
    :param img:
    :param roi:
    :param scale:
    :return:
    '''
    h, w = img.shape
    if roi is None:
        roi = np.ones((h, w), dtype=np.uint8)
    h_s, w_s = int(h * scale), int(w * scale)
    img_s = cv2.resize(img, (w_s, h_s))
    mask_s = cv2.resize(roi, (w_s, h_s))
    return np.median(img_s[mask_s>0])

if __name__ == '__main__':
    # BG_table_path = '../meta/BG_table.csv'
    # BH_factor_path = '../meta/BH_factor.csv'
    # load_bk_params(BG_table_path, BH_factor_path, kvp=30, mas=100)
    a = 1
