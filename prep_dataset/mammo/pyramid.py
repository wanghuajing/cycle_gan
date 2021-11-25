import numpy as np
import cv2

def pyramid_decompose(img, n_level=8):
    '''
    对图像进行laplacian金字塔分解
    :param img:
    :param n_level: 金字塔层数，包括n_level - 1层细节图和1层底层高斯图，
    :return: laplacian_pyramid：一个长度为n_level的list. laplacian_pyramid[n_level]是最底层高斯图，
    laplacian_pyramid[i] 是第i层细节图，i=0为最高频，i=n_level-1是最低频。
    '''
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(n_level):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
    # Laplacian Pyramid
    layer = gaussian_pyramid[n_level-2]
    laplacian_pyramid = [layer]
    for i in range(n_level-1, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    # laplacian_pyramid.reverse()
    return laplacian_pyramid

def pyramid_decompose_full(img, n_level=8):
    '''
    对图像进行laplacian金字塔分解,但分解后每层频率子图像仍然为原始图像大小
    :param img:
    :param n_level: 金字塔层数，包括n_level - 1层细节图和1层底层高斯图，
    :return: laplacian_pyramid：一个长度为n_level的list. laplacian_pyramid[n_level]是最底层高斯图，
    laplacian_pyramid[i] 是第i层细节图，i=0为最高频，i=n_level-1是最低频。
    '''
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(n_level):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
    # Laplacian Pyramid
    layer = gaussian_pyramid[n_level-2]
    laplacian_pyramid = [layer]
    for i in range(n_level-1, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    # 遍历底层高斯图和每一层细节图，通过PyrUp恢复至原始分辨率。注意此处不可直接一次上采样至原始分辨率。
    laplacian_pyramid_full = laplacian_pyramid[0:2]
    for i in range(1, n_level-1):
        size = (laplacian_pyramid[i + 1].shape[1], laplacian_pyramid[i + 1].shape[0])
        for j, im in enumerate(laplacian_pyramid_full):
            laplacian_pyramid_full[j] = cv2.pyrUp(im, dstsize=size)
        laplacian_pyramid_full.append(laplacian_pyramid[i+1])
    return laplacian_pyramid_full