import cv2
import os
from tqdm import tqdm

path = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/full_cut20/add/'
path1 = '/media/zhao/HD1/data/ai-postprocess/mammo300_png/full_cut20/add_1/'
lists = os.listdir(path)
for name in tqdm(lists):
    img = cv2.imread(path + name, -1)
    img = cv2.resize(img, (3328, 4056), cv2.INTER_CUBIC)
    cv2.imwrite(path1 + name, img)
