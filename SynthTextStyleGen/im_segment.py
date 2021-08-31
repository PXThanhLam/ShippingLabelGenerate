from __future__ import division
from skimage.measure import label
from skimage.segmentation import slic,felzenszwalb,quickshift,watershed
from skimage.util import img_as_float
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import cv2
import h5py
import numpy as np
import multiprocessing as mp
import traceback, sys
from collections import Counter
import sys
import os
from tqdm import tqdm
from skimage.filters import sobel


root = 'depth_segm/input'
for im_path in tqdm(os.listdir(root)):
  image = cv2.imread(root + '/' + im_path)
  bgr_h,bgr_w,_ = image.shape
  big_size = 1200
  image = cv2.resize(image,(int((big_size*bgr_w)/max(bgr_h,bgr_w)),int((big_size*bgr_h)/max(bgr_h,bgr_w))))
  # depth = cv2.imread('depth_segm' + '/output_monodepth/' + im_path)
  # image = np.array(0.8*image + 0.2*depth, dtype=np.uint8)
  # cv2.imwrite('depth_segm/output_semseg/depth_im_' + im_path,image)
  segments = felzenszwalb(img_as_float(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)), scale=int(image.shape[0]*image.shape[1]/1000),
                          sigma=0.5, min_size=int(image.shape[0]*image.shape[1]/400))
  

  np.save('depth_segm/output_semseg/' + im_path.split('.')[0]+'.npy',segments)

  count_regions = Counter(segments.reshape(segments.shape[1]*segments.shape[0]))
 
  regions_color = {}
  for k in count_regions.keys():
    regions_color[k] = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
  for i in range(len(image)):
    for j in range(len(image[0])):
      image[i][j] = image[i][j] * 0.2 + regions_color[segments[i][j]]*0.8
  cv2.imwrite('depth_segm/output_semseg/' + im_path,image)