from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
from skimage.util import dtype
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math
from common import *
import pickle
import sys
sys.path.append('../NaiveGen')
from augment_util_2d import *

eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
eig_vec = np.array([
	[-0.58752847, -0.69563484, 0.41340352],
	[-0.5832747, 0.00994535, -0.81221408],
	[-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)
data_rng = np.random.RandomState(123)

class RenderForm(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir):
        self.f_shrink = 0.90
        self.max_trials = 2
        
        self.min_form_h = 40
        self.p_flat = 0.10


        self.form_state = FormState(data_dir)

        pygame.init()



    def place_form(self, form, back_arr):
        h,w,_ = form.shape
        locs = [None]
        form_mask = np.zeros_like(back_arr, dtype=np.float32)
        form_img = np.zeros((form_mask.shape[0],form_mask.shape[1],3))
        
        back_h, back_w = back_arr.shape
        if h>= back_h or w >= back_w:
            return None,None,None
        # bbs = np.zeros((2,4,1))
        # bbs[:,0,0] = [loc[0],loc[1]]
        # bbs[:,1,0] = [loc[0] + w, loc[1]]
        # bbs[:,2,0] = [loc[0] + w, loc[1] + h]
        # bbs[:,3,0] = [loc[0],loc[1] + h]



        # blit the text onto the canvas
        bgr = np.zeros((h * 3, w * 3, 3))
        bgr[h : 2*h, w : 2*w, : ] = form
        fgr_img = bgr
        st_border = 5
        mask = np.zeros((h * 3, w * 3))
        mask[h + st_border : 2*h - st_border, w + st_border : 2*w - st_border ] = 1
        keypoints = [[w + st_border, h + st_border],[2*w - st_border, h + st_border],
                      [w + st_border, 2*h -st_border],[2*w -st_border, 2*h - st_border]]
        keypoints = np.array(keypoints, dtype=np.float32)
        fgr_img, mask, keypoints = foreground_random_transform(fgr_img, data_rng, eig_val, eig_vec, mask, degree = (-30,30), keypoint=keypoints)
        i, j = np.where(mask)
        if len(i) == 0 or len(j) == 0 :
            return None,None,None
        min_i, max_i, min_j, max_j = min(i), max(i), min(j), max(j)
        indices = np.meshgrid(np.arange(min_i, max_i + 1), np.arange(min_j, max_j + 1), indexing='ij')
        fgr_img = fgr_img[tuple(indices)]
        mask = mask[tuple(indices)]
        keypoints = keypoints - np.array([min_j,min_i])
        
        
        st_border = 1
        h,w,_ = fgr_img.shape
        if h >= back_h -st_border or w >= back_w -st_border :
            return None,None,None
        
        loc = [None,None]
        loc[1] = np.random.randint(0,back_h-h)
        loc[0] = np.random.randint(0,back_w-w)
        
        form_mask[loc[1] +st_border : loc[1]+h -st_border, loc[0]  +st_border :loc[0]+w  -st_border ] += mask[st_border:h-st_border,st_border:w-st_border]
        form_img[ loc[1] +st_border : loc[1]+h -st_border, loc[0]  +st_border :loc[0]+w  -st_border:] = fgr_img[st_border:h-st_border,st_border:w-st_border]
        keypoints = keypoints + np.array([loc[0],loc[1]])
        
        # keypoints = np.reshape(np.array([keypoints]),(2,4,1))
        # bbs = np.reshape(np.array([bbs]),(2,4,1))
        # print(keypoints)
        # result_img = form_img.copy()
        # for point in np.reshape(keypoints[:,:,0],(4,2)):
        #     print(point)
        #     rand_color = [245,34,210]
        #     result_img = cv2.circle(result_img, center= (int(point[0]), int(point[1])), radius= 5, color = rand_color , thickness= -1)
        # cv2.imwrite('debug.png',result_img)
        # print('------')            

        print(form.shape)

        return form_mask, form_img,keypoints

    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m,axis=0))
        rW = np.median(np.sum(m,axis=1))
        return rH,rW

    def sample_form_height_px(self,h_min,h_max,num_blend_form):
        max_big_size = h_max // ( 1 + num_blend_form//5) - 10
        min_big_size = max(h_min,h_max // (num_blend_form * 0.8))
        if min_big_size > max_big_size:
            return None
        return rand_int_gaussian(min_big_size,max_big_size,(3 * max_big_size + 2 * min_big_size)//5)

    def bb_xywh2coords(self,bbs):
        coords = np.zeros((2,4))
        coords[:,:] = bbs[:2]
        coords[0,1] += bbs[2]
        coords[:,2] += bbs[2:4]
        coords[1,3] += bbs[3]
        return coords


    def render_sample(self,form,mask,num_blend_form):

        H,W = self.robust_HW(mask)
        f_asp = self.form_state.get_aspect_ratio(form)

        max_form_h = int(0.95*H)
        if max_form_h < self.min_form_h: 
            return 
        i = 0
        while i < self.max_trials and max_form_h > self.min_form_h:
            f_h = self.sample_form_height_px(self.min_form_h, max_form_h,num_blend_form)
            if f_h is None:
                return
            while f_h < 70 and i<=4:
                f_h = self.sample_form_height_px(self.min_form_h, max_form_h,num_blend_form)
                i +=1
            i += 1      
            if f_h < 45 :
                if np.random.rand() > f_h/100 :
                    return 
            if i >= self.max_trials:
                break       
            max_form_h = f_h 
            f_w =  int(f_h * form.shape[1] / form.shape[0])
            # if f_h < 60 or max(f_h/f_w,f_w/f_h) > 1.6:
            #     return
            form = cv2.resize(form,(int(f_w),int(f_h)))
           
            if np.any(np.r_[form.shape[:2]] > np.r_[mask.shape[:2]]):
                continue

            form_mask,form_img,keypoint = self.place_form(form, mask)
            if form_mask is not None :
                return form_mask,form_img,keypoint
        return 


    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(ta, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=128, thickness=1)
        plt.imshow(ta,cmap='gray')
        plt.show()


class FormState(object):
    """
    Defines the random state of the font rendering  
    """
    def __init__(self, data_dir):
        self.form_dir = data_dir
        self.all_form_img = np.array([self.form_dir + '/' + path for path in os.listdir(self.form_dir)])
    def sample(self):
        select_foreground = self.all_form_img[np.random.randint(0,len(self.all_form_img) - 1, size = 1)][0]
        return cv2.cvtColor(cv2.imread(select_foreground), cv2.COLOR_RGB2BGR)
    def get_aspect_ratio(self, form, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        h, w, _ = form.shape
        return w/h

    

    