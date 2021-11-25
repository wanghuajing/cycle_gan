import cv2
import numpy as np

split_width = 512
split_height = 512


def start_points(size, split_size, overlap=128):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def split_img(img, split_width=512, split_height=512, overlap=128):
    img_h, img_w = img.shape
    count = 0
    X_points = start_points(img_w, split_width, 0.5)
    Y_points = start_points(img_h, split_height, 0.5)
    split_stack = np.zeros(split_height, split_width, len(X_points)*len(Y_points))
    for i in Y_points:
        for j in X_points:
            split_stack[:,:,count] = img[i:i + split_height, j:j + split_width]
            count += 1
    return split_stack




