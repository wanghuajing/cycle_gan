import numpy as np
import cv2
from skimage.measure import label

def getLargestCC(img):
    '''
    分割乳房主体区域的函数，基于otsu分割。该函数仍然有待改进，主要是要让边界更平滑，同时能处理多个不连通的物体。
    :param img: image in float type with in [0,1]
    :return: 和img等大的图, 0代表背景,1代表乳房区域
    '''
    mask = np.ones(img.shape, dtype=np.uint8)
    i_min, i_max = np.percentile(img, [1, 99])
    img = (img - i_min) / (i_max - i_min)
    img[img > 1] = 1
    img[img < 0] = 0
    img_8bit = (img * 255).astype(np.uint8) # opencv的otsu分水岭算法只支持uint8格式
    retval, thresh_gray = cv2.threshold(img_8bit, 0, 2**8-1, \
                                       type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labels = label(thresh_gray) # Label connected regions of an integer array.
    assert( labels.max() != 0 ) # assume at least 1 CC
    # np.bincount: Count number of occurrences of each value in array of non-negative ints.
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = largestCC.astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    largestCC =  cv2.morphologyEx(largestCC,cv2.MORPH_CLOSE,kernel)
    return largestCC

def segbreast_cv2(img, method='otsu'):
    '''
    乳房分割的函数。输出全部大于分割阈值的fgmask和最大联通区域bmask
    :param img: 反图。即之谜部分亮度高。如果使用otsu方法，则输入为原始图的反图。如果使用triangle方法，输入为log(od)图的反图
    :param method: otsu或者triangle。
    :return: fgmask, 所有大于阈值部分的mask。 bmask，基于fgmask只保留最大联通区域。
    '''
    img = img.astype(np.float32)
    # 去掉可能的奇异点，归一化到0-1之间
    i_min, i_max = np.percentile(img, [1,99])
    img = (img - i_min)/(i_max - i_min)
    img[img>1] = 1
    img[img<0] = 0

    # opencv的otsu分水岭算法只支持uint8格式，转换为8bit。
    img_8bit = (img * 255).astype(np.uint8)
    if method == 'otsu':
        thresh, fgmask = cv2.threshold(img_8bit, 0, 1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == 'triangle':
        thresh, fgmask = cv2.threshold(img_8bit, 0, 1, type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # 对fgmask进行处理平滑，去掉可能的空洞
    # kernel = np.ones((11, 11), np.uint8)
    # 用一个原型的kernel可能会好一些
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # 再做open可以消除边缘毛刺，感觉做不做都行。
    #
    labels = label(fgmask) # Label connected regions of an integer array.
    assert( labels.max() != 0 ) # assume at least 1 CC
    # np.bincount: Count number of occurrences of each value in array of non-negative ints.
    bmask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    bmask = bmask.astype(np.uint8)
    return fgmask, bmask

def bimodalTest(y):
    '''
    判断直方图y是否bimodal: 有且仅有两个局部峰值
    :param y:
    :return:
    '''
    ylen = len(y)
    is_bimodal = False
    modes = 0 # 峰值数量
    for k in range(1, ylen-1):
        if ((y[k-1]<y[k]) and (y[k+1] < y[k]) ): # 局部峰值
            modes += 1
            if modes > 2:
                is_bimodal = False
                break
    if modes == 2:
        is_bimodal = True
    return is_bimodal

def seghist_prewitt(iHisto):
    '''
    基于直方图的分割方法。该分割方法假设图像主要由前景背景两部分组成。主要思路是，
    1. 判断直方图是否bi-modal, 即只有两个局部最高点
    2. 如果是，取两峰值之间的最低点作为分割阈值
    3. 如果不是，对直方图进行长度为3的均值平滑，返回1.
    :param iHisto:
    :return:
    '''
    iter = 0
    threshold = -1
    converged = True
    iHisto = iHisto.astype(np.float) # 转为浮点数，因为之后要进行平均等操作
    tHisto = np.zeros(len(iHisto)) # 临时直方图变量
    while (not bimodalTest(iHisto)): # 1. 判断是否bimodal, 如果不是则进入循环
        for i in range(1, 255): # 邻域为3的均值平滑
            tHisto[i] = (iHisto[i-1] + iHisto[i] + iHisto[i+1])/3
        tHisto[0] = (iHisto[0] + iHisto[1])/3 # 处理下边界量
        tHisto[255] = (iHisto[254] + iHisto[255])/3 # 处理上边界两
        iHisto = tHisto.copy() # 将临时变量赋值给直方图，进行下一次循环
        iter += 1
        if (iter>10000):
            converged = False # 在10000个循环内无法变成bimodal
            print('fail to find threshold in 10000 iterations')
    if converged:
        for i in range(1, 255): # 找到两个峰值之间的局部最低点，作为阈值
            if ((iHisto[i-1] > iHisto[i]) and (iHisto[i+1]>=iHisto[i])):
                threshold = i
                break
    return threshold

def segbreast_prewitt(img):
    '''
    乳房分割的函数。输出全部大于分割阈值的fgmask和最大联通区域bmask
    :param img: 反图。即之谜部分亮度高。如果使用otsu方法，则输入为原始图的反图。如果使用triangle方法，输入为log(od)图的反图
    :param method: otsu或者triangle。
    :return: fgmask, 所有大于阈值部分的mask。 bmask，基于fgmask只保留最大联通区域。
    '''
    img = img.astype(np.float32)
    # 去掉可能的奇异点，归一化到0-1之间
    i_min, i_max = np.percentile(img, [1,99])
    img = (img - i_min)/(i_max - i_min)
    img[img>1] = 1
    img[img<0] = 0


    img_8bit = (img * 255).astype(np.uint8)
    hist = cv2.calcHist([img_8bit], [0], None, [256], [0, 256]) # 计算256位直方图
    thresh = seghist_prewitt(hist) # 调用阈值函数
    if thresh == -1:
        print('min segmentation failed, use otsu segmentation instead')
        thresh, fgmask = cv2.threshold(img_8bit, 0, 1, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        fgmask = (img_8bit>thresh).astype(np.uint8)

    # 对fgmask进行处理平滑，去掉可能的空洞
    # kernel = np.ones((11, 11), np.uint8)
    # 用一个原型的kernel可能会好一些
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 再做open可以消除边缘毛刺，感觉做不做都行。
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    labels = label(fgmask) # Label connected regions of an integer array.
    assert( labels.max() != 0 ) # assume at least 1 CC
    # np.bincount: Count number of occurrences of each value in array of non-negative ints.
    bmask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    bmask = bmask.astype(np.uint8)
    return fgmask, bmask

def segbreast_refine(img, method='otsu', bk_std_thresh=5):
    '''
    乳房分割的函数。输出全部大于分割阈值的fgmask和最大联通区域bmask
    :param img: 反图。即之谜部分亮度高。如果使用otsu方法，则输入为原始图的反图。如果使用triangle方法，输入为log(od)图的反图
    :param method: otsu或者triangle。
    :return: fgmask, 所有大于阈值部分的mask。 bmask，基于fgmask只保留最大联通区域。
    '''
    img = img.astype(np.float32)
    # 去掉可能的奇异点，归一化到0-1之间
    i_min, i_max = np.percentile(img, [1,99])
    img = (img - i_min)/(i_max - i_min)
    img[img>1] = 1
    img[img<0] = 0

    # img = cv2.medianBlur(img, 3)
    # opencv的otsu分水岭算法只支持uint8格式，转换为8bit。
    img_8bit = (img * 255).astype(np.uint8)
    if method == 'otsu':
        thresh, fgmask = cv2.threshold(img_8bit, 0, 1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == 'triangle':
        thresh, fgmask = cv2.threshold(img_8bit, 0, 1, type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    pix_bk = img[(1-fgmask).astype(bool)] # 取出第一次分割的bk像素值
    # pix_bk = pix_bk[pix_bk<np.percentile(pix_bk, 99)]
    std_bk = np.std(pix_bk)
    mean_bk = np.mean(pix_bk)
    thresh_1 = mean_bk + bk_std_thresh*std_bk
    thresh_final = thresh_1 if thresh_1 < thresh/255. else thresh/255.
    fgmask = (img>thresh_final).astype(np.uint8)

    # 对fgmask进行处理平滑，去掉可能的空洞
    # kernel = np.ones((11, 11), np.uint8)
    # 用一个原型的kernel可能会好一些
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)  # 这里是新加的， open帮助平滑边缘，以及去掉背景的分割噪点。

    # 再做open可以消除边缘毛刺，感觉做不做都行。
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    labels = label(fgmask) # Label connected regions of an integer array.
    assert( labels.max() != 0 ) # assume at least 1 CC
    # np.bincount: Count number of occurrences of each value in array of non-negative ints.
    bmask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    bmask = bmask.astype(np.uint8)
    return fgmask, bmask


def get_cmask(bmask):
    bmask = bmask.astype(np.float32)
    a = np.sum(bmask, axis=1)
    y = np.where(a>150)
    if len(y[0]) > 1000: # 赵：20210831新增。防止拍摄异常情况下， 例如广州 易江 7号图，上图错误， bmask面积非常小，是的上一步得到的y为空值，执行下面的语句报错。
        y_max = np.max(y)
        y_min = int(y_max*0.33)
        # y_h = 1020
        bmask[0:y_min,:] = 0
        bmask[y_max:,:] = 0
    return bmask.astype(np.uint8)

def get_vmask(bmask, ismlo):
    '''
    从bmask获取vmask，即有效亮度控制范围。主要是去除乳房边缘的一部分像素，这一部分像素由于皮肤的过度增强，经常导致异常亮度。
    :param bmask:
    :param ismlo:
    :return:
    '''
    bmask = bmask.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bmask = cv2.erode(bmask, kernel, iterations=20) # erode 20次，相当于向内腐蚀100像素
    a = np.sum(bmask, axis=1) # 向胸壁测投影
    y = np.where(a>50) # x方向累计超过50像素为非0的区域

    if ismlo:
        y_min = 500 # 对于MLO位固定去掉最上面的500像素。这一部分经常包含皮肤褶皱带来的异常亮度。
    else:
        y_min = np.min(y)
    y_max = np.max(y)
    bmask[:y_min,:] = 0
    bmask[y_max:, :] = 0
    return bmask.astype(np.uint8)