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
from tqdm import tqdm
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
    return cv2.seamlessClone(paste_img, image_patch, paste_mask, center, cv2.NORMAL_CLONE)

fore_ground_root = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/Foreground_Extract/'
fgr_same_border = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/Foreground_same_border/'
all_fgr_img_path = [fore_ground_root + p for p in os.listdir(fore_ground_root)]
num_gen = 1200
idx = 0
for _ in tqdm(range(num_gen)) :
    path = all_fgr_img_path[np.random.randint(0,len(all_fgr_img_path))]
    img = cv2.imread(path)
    im_h, im_w, _ = img.shape
    if im_h * im_w >= 1000 * 1000 :
        continue
    up_scale_range = np.random.uniform(1.05, 1.15)
    white_mask_h, white_mask_w = int(im_h * up_scale_range), int(im_w * up_scale_range)
    white_img = np.ones((white_mask_h, white_mask_w)) * 255 
    white_img = np.stack([white_img,white_img,white_img]).transpose(1,2,0)
    mask = np.zeros((white_mask_h, white_mask_w))
    st_x, st_y = (white_mask_w - im_w)// 2,(white_mask_h - im_h)// 2

    white_img[st_y : st_y + im_h, st_x : st_x + im_w] = poisson_copy_paste(white_img[st_y : st_y + im_h, st_x : st_x + im_w],
                                                                          img, white_img[st_y : st_y + im_h, st_x : st_x + im_w][:,:,0]// 255) 

    cv2.imwrite(fgr_same_border + str(idx) +'.png', white_img)
    idx += 1