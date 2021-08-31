from numpy import ma
from numpy.core.fromnumeric import resize
from numpy.lib.function_base import copy
import scipy.io
import cv2
import os
import numpy as np
from skimage.util import dtype
from augment_util_2d import foreground_random_transform
import time
from skimage.filters import gaussian
import random
import json
from pycocotools import mask as pycoco_mask
from itertools import groupby
import math
def get_all_imgpath(root):
    return [root + '/' + path for path in os.listdir(root)]

def rand_int_gaussian(low,high,mean):
    sigma = min(mean-low,high-mean) 
    rand_int = np.random.normal(mean,sigma)
    if rand_int < low:
        rand_int = int(np.random.uniform(low, mean))
    return int(np.clip(rand_int, low,high))
def copy_paste(image_patch, paste_img, paste_mask):
    paste_mask = gaussian(paste_mask, sigma=1.0, preserve_range=True)
    paste_mask = np.stack([paste_mask,paste_mask,paste_mask]).transpose(1,2,0)

    return image_patch * (1 - paste_mask) + paste_img * paste_mask
def poisson_copy_paste(image_patch, paste_img, paste_mask):
    center = (image_patch.shape[1]// 2, image_patch.shape[0] // 2)
    paste_mask = np.stack([paste_mask,paste_mask,paste_mask]).transpose(1,2,0)

    paste_mask = np.array(paste_mask * 255 , dtype = np.uint8)
    paste_img = np.array(paste_img, dtype = np.uint8)
    image_patch = np.array(image_patch, dtype = np.uint8)
    return cv2.seamlessClone(paste_img, image_patch, paste_mask, center, cv2.MIXED_CLONE)
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
def gen_layout(width, height, num, stx = 0, sty = 0):
    assert num >=1
    if num == 1:
        return [(stx,sty,stx + width, sty + height)]
    mean_num_split = math.ceil(np.sqrt(num))
    # num_split = np.random.normal(mean_num_split, mean_num_split / 3)
    # num_split = int(np.clip(num_split,2,max(2,mean_num_split)))
    num_split = min(num,max(mean_num_split, int(max(width,height)/min(width,height))))
    num_per_split = (math.floor(num/num_split),math.ceil(num/num_split))
    w_h_ratio  = width / height
    prob = 1 - math.pow(2,- w_h_ratio)
    layout_res = []
    #||
    if np.random.normal(0.5,0.15) < prob:
        for i in range(num_split):
            if i!= num_split - 1 :
                num_gen = num_per_split[0] if np.random.uniform() >=0.5 else num_per_split[1]
            else:
                num_gen = max(1,num - len(layout_res))
            stx_this_level, sty_this_level = stx + i * width//num_split, sty
            layout_this_level = gen_layout(width//num_split, height,num_gen, stx_this_level, sty_this_level)
            layout_res.extend(layout_this_level)
    #---
    else:
        for i in range(num_split):
            if i!= num_split - 1 :
                num_gen = num_per_split[0] if np.random.uniform() >=0.5 else num_per_split[1]
            else:
                num_gen =  max(1,num - len(layout_res))
            stx_this_level, sty_this_level =  stx, i * height//num_split + sty
            layout_this_level = gen_layout(width, height//num_split, num_gen, stx_this_level, sty_this_level)
            layout_res.extend(layout_this_level)
    
    return layout_res

       

# w,h = 700,900
# for i in range(10):
#     layout = gen_layout(w,h,14)
#     print(layout)
#     bgr = np.zeros((h,w,3))
#     for i in layout :
#         rand_color =(np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255))
#         cv2.rectangle(bgr, (i[0],i[1]), (i[2], i[3]),rand_color, -1)
#     cv2.imwrite('im.png',bgr)
