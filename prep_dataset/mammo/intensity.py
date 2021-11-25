import numpy as np
from matplotlib import pyplot as plt

def compute_od(img_raw, largestCC, od_max=3.2):
    '''
    粗略计算optical density, 即相对曝光剂量. od = np.log10((img_bk+1)/(img_raw+1)).
    但是我们现在没有空拍的bk图，因此只能取背景像素亮度值的中位数作为bk
    :param img_raw: 原始图
    :param largestCC: 乳房遮罩。乳房内部，largestCC=1
    :param od_max: # 设置最大的optical density. 超过这个数一般没有意义。这个数大致是从Hologic的后处理文本段提取出来的。
    :return:
    '''
    bk_median = np.median(img_raw[largestCC == 0])
    # 计算吸光度图
    img_od = np.log10(bk_median + 1) - np.log10(img_raw + 1)  # 吸光度图
    img_od[img_od > od_max] = od_max
    img_od[img_od < 0] = 0
    return img_od

def load_bk_params(BG_table_path, BH_factor_path, kvp, mas):
    BG_table = np.genfromtxt(BG_table_path, delimiter=',')
    BH_factor = np.genfromtxt(BH_factor_path,delimiter=',')
    bg = mas*BG_table[BG_table[:,0]==kvp, 1][0]
    p = np.zeros(4)
    p[1:4] = BH_factor[BH_factor[:,0]==kvp, 1:].squeeze()
    return bg, p

def compute_od_correct(img, bg, p=None, correct=False):
    img_att = np.log(bg+1) - np.log(img+1)
    if correct:
        img_BH = np.polyval(p, img_att)
    else:
        img_BH = img_att
    img_BH = img_BH / np.log(10)
    img_BH[img_BH<0] = 0
    return img_BH

def range_compress(img, largestCC, r=[0.05, 0.05], p=[0.01, 0.001], debug=False):
    '''
    压缩img的动态范围。
    该函数使乳房遮罩(largestCC)内的亮度值最低的p[0]比例的像素，占用不超过r[0]的动态范围。
    亮度值最高的p[1]比例的像素，占用不超过r[1]的动态范围。
    :param img:
    :param largestCC:
    :param r:
    :param p:
    :param show:
    :return:
    '''
    # 首先按照乳房內部亮度进行最大最小归一化

    if debug:
        plt.hist(img[largestCC>0], bins=1000)
        plt.title('hist before compression')
        plt.show()

    t_low = np.percentile(img[largestCC>0], p[0]*100)
    t_high = np.percentile(img[largestCC>0], (1-p[1])*100)

    t_low_new = np.minimum(t_low, r[0])
    t_high_new = np.maximum(t_high, 1-r[1])

    landmarks = np.array([0, t_low, t_high, 1])
    landmarks_new = np.array([0, t_low_new, t_high_new, 1])

    # 线性插值
    img_new = img.copy()
    img_new = np.interp(img_new, landmarks, landmarks_new)
    # 乳房内部亮度值已经归一化到0-1区间， 乳房遮罩外部亮度可能大于0或者小于1， 需要处理一下。
    img_new[img_new>1] = 1
    img_new[img_new<0] = 0

    if debug:
        plt.hist(img_new[largestCC>0], bins=1000)
        plt.title('hist after compression')
        plt.show()

    return img_new

def img_normalize(img, clip = []):
    '''
    将图像归一化到0-1之间的函数
    :param img:
    :param clip: 如果提供，则是设定的最小值和最大值。否则默认以整幅图像的最大值和最小值判断。
    :return:
    '''
    if clip:
        clip_min, clip_max = clip[0], clip[1]
    else:
        clip_min, clip_max = img.min(), img.max()
    img[img>clip_max] = clip_max
    img[img<clip_min] = clip_min
    img = (img - clip_min)/(clip_max - clip_min)
    return img

def soft_normalize(img, mask=None, params=None):
    '''
    图像标准化，参见：
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
    中非线性标准化部分。
    :return:
    '''
    if params is None:
        p_p = [0.01, 0.99]
        p_alpha = 4
        p_beta =  0.17
    else:
        p_p = params['p']
        p_alpha = params['alpha']
        p_beta = params['beta']

    p_D_Max = 65535
    img = img*p_D_Max
    pix_valid = img[mask>0]

    t_low = np.percentile(pix_valid, p_p[0] * 100)
    t_high = np.percentile(pix_valid, p_p[1] * 100)
    WW = t_high - t_low
    WC = (t_high + t_low)/2

    VOILUT = VOI_LUT(WW, WC, p_alpha, p_beta, p_D_Max)
    print('voi-lut mapping min: %.3f max: %.3f'%(VOILUT[int(t_low)].astype(float)/p_D_Max, VOILUT[int(t_high)].astype(float)/p_D_Max))

    # plt.figure()
    # plt.plot(VOILUT)
    # plt.grid()
    # plt.title("alpha=%.2f beta=%.2f"%(p_alpha, p_beta))
    # plt.show()
    img = img.astype(np.uint16)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i,j] = VOILUT[img[i,j]]
    img = VOILUT[img]

    # plt.figure()
    # plt.imshow(img,cmap='gray')
    # plt.show()
    img = img.astype(np.float32) / p_D_Max

    return img

def soft_normalize_1(img, thresh, mask=None, params=None):
    '''
    图像标准化，参见：
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
    中非线性标准化部分。
    :return:
    '''
    if params is None:
        p_p = [0.01, 0.99]
        p_alpha = 4
        p_beta =  0.17
    else:
        p_p = params['p']
        p_alpha = params['alpha']
        p_beta = params['beta']

    idx_fg = mask>0

    abs_max_pos = img[idx_fg].max()
    abs_max_neg = - img[idx_fg].min()
    if abs_max_pos > abs_max_neg:
        abs_max = abs_max_pos
    else:
        abs_max = abs_max_neg

    img = img/abs_max + 0.5
    p_D_Max = 65535
    img = img*p_D_Max

    t_low = (0.5 - thresh/abs_max)*65535
    t_high = (0.5 + thresh/abs_max)*65535
    WW = t_high - t_low
    WC = (t_high + t_low)/2

    VOILUT = VOI_LUT(WW, WC, p_alpha, p_beta, p_D_Max)

    # plt.figure()
    # plt.plot(VOILUT)
    # plt.grid()
    # plt.title("alpha=%.2f beta=%.2f"%(p_alpha, p_beta))
    # plt.show()
    img = img.astype(np.uint16)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i,j] = VOILUT[img[i,j]]
    img = VOILUT[img]

    # plt.figure()
    # plt.imshow(img,cmap='gray')
    # plt.show()
    img = img.astype(np.float32) / p_D_Max
    img = (img - 0.5)*abs_max
    return img

def soft_normalize_3(img, low, high, mask=None, params=None):
    '''
    图像标准化，参见：
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
    中非线性标准化部分。
    :return:
    '''
    if params is None:
        p_p = [0.01, 0.99]
        p_alpha = 4
        p_beta =  0.17
    else:
        p_p = params['p']
        p_alpha = params['alpha']
        p_beta = params['beta']

    idx_fg = mask>0

    img_max = np.max(img[idx_fg])
    img_min = np.min(img[idx_fg])

    img = (img - img_min)/(img_max - img_min)
    p_D_Max = 65535
    img = img*p_D_Max

    t_low = ((low-img_min)/(img_max - img_min))*65535
    t_high = (high - img_min)/(img_max - img_min)*65535
    WW = t_high - t_low
    WC = (t_high + t_low)/2

    VOILUT = VOI_LUT(WW, WC, p_alpha, p_beta, p_D_Max)
    print(VOILUT[int(t_low)]/65535.)
    print(VOILUT[int(t_high)]/65535)

    # plt.figure()
    # plt.plot(VOILUT)
    # plt.grid()
    # plt.title("alpha=%.2f beta=%.2f"%(p_alpha, p_beta))
    # plt.show()
    img = img.astype(np.uint16)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i,j] = VOILUT[img[i,j]]
    img = VOILUT[img]

    # plt.figure()
    # plt.imshow(img,cmap='gray')
    # plt.show()
    img = img.astype(np.float32) / p_D_Max
    return img

def VOI_LUT(WW, WC, alpha, beta, D_Max):

    VOILUT = np.zeros(D_Max + 1)
    if np.abs(beta) > 0 and np.abs(beta) <= 0.01:

        Magnitude_beta = 0.01

    elif np.abs(beta) > 0.01 and  np.abs(beta) <= 1:

        Magnitude_beta = abs(beta)

    else:
        print('beta is not a valid number')

    dx = WW / (alpha * Magnitude_beta * D_Max ** Magnitude_beta)

    if beta < 0:
    # When beta is negative, the curve will exhibit high contrast in the toe of the curve.Construct the LUT in the following manner:
        threshold = WC - WW / (alpha * Magnitude_beta) + dx

        for i in range(D_Max + 1):
            x = i   # x - [0, 65535]
            if x < threshold:
               VOILUT[i] = 0
            else:
               temp = (1 + (alpha * Magnitude_beta / WW) * (x - WC)) ** (1 / Magnitude_beta)
               VOILUT[i] = D_Max - np.fix((D_Max + 1) / (1 + temp))
    else:
        threshold = WC + WW / (alpha * Magnitude_beta) - dx

        for i in range(D_Max + 1):
             x = i   # x - [0, 65535]
             if x > threshold:
                VOILUT[i] = D_Max
             else:
                temp = (1 + ((alpha * Magnitude_beta) / WW) * (WC - x))** (1 / Magnitude_beta)
                VOILUT[i] = np.fix((D_Max + 1) / (1 + temp))

    return VOILUT

def soft_normalize_2(img, thresh, mask=None, params=None):
    '''
    图像标准化，参见：
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
    中非线性标准化部分。
    :return:
    '''
    if params is None:
        p_p = [0.01, 0.99]
        p_alpha = 4
        p_beta =  0.17
    else:
        p_p = params['p']
        p_alpha = params['alpha']
        p_beta = params['beta']

    idx_fg = mask>0

    abs_max_pos = img[idx_fg].max()
    abs_max_neg = - img[idx_fg].min()
    if abs_max_pos > abs_max_neg:
        abs_max = abs_max_pos
    else:
        abs_max = abs_max_neg

    img = img/abs_max

    t_low = - thresh/abs_max
    t_high = thresh/abs_max
    WW = t_high - t_low
    WC = 0

    img = soft_win(img, WW, WC, p_alpha, p_beta)
    n_high = soft_win(t_high,WW, WC, p_alpha, p_beta)
    img = img*(t_high/n_high)*abs_max

    return img

def soft_win(x, WW, WC, alpha, beta):
    if np.abs(beta) > 0 and np.abs(beta) <= 0.01:
        Magnitude_beta = 0.01
    elif np.abs(beta) > 0.01 and  np.abs(beta) <= 1:
        Magnitude_beta = abs(beta)
    else:
        print('beta is not a valid number')
    # When beta is negative, the curve will exhibit high contrast in the toe of the curve.Construct the LUT in the following manner:
    temp = (1 + (alpha * Magnitude_beta / WW) * (x - WC)) ** (1 / Magnitude_beta)
    if beta <0:
        y = 1 - (1+1)/(temp + 1)
    else:
        y = (1+1)/(temp + 1)
    return y

def gen_thickness(img, md, t):
    # t = [0.8, 0.9, 1.0, 1.1, 1,2]
    img_is = np.zeros((img.shape[0], img.shape[1], len(t)))
    for i in range(len(t)):
        img_tmp = img.copy()
        img_tmp[img_tmp>t[i]*md] = t[i]*md
        img_tmp = img_tmp/(t[i]*md)
        img_is[:,:,i] = img_tmp
    img_t = np.mean(img_is, axis=2)
    return img_t

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

def sigmoid_w(x, wc, ww):
    return 1/(1+np.exp(-4*(x-wc)/ww))

def make_sigmoid_lut(wc, ww, input_range, out_max):
    lut = np.zeros(input_range)
    for idx in range(input_range):
        lut[idx] = out_max*sigmoid_w(idx, wc, ww)
    lut.astype(np.uint16)
    return lut