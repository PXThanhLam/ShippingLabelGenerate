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
from gen_data_utils import *
from tqdm import tqdm
def update_mask(masks, paste_mask):
    masks = [
        np.logical_and(mask, np.logical_xor(mask, paste_mask)).astype(np.uint8) for mask in masks
    ]
    masks.append(paste_mask)
    return masks
def update_keypoint(all_keypoints, new_keypoint, new_masks):
    h, w = new_masks.shape
    for kp in all_keypoints:
        new_vis = []
        for point,vis in zip(kp['points'], kp['visibility']) :
            if point[0] < 0 or point[0] >= w  or point[1] < 0 or point[1] >= h :
                new_vis.append(0)
            elif new_masks[int(point[1])][int(point[0])] == 1:
                new_vis.append(0)
            else:
                new_vis.append(vis)
        kp['visibility'] = new_vis
    vis = []
    for point in new_keypoint :
        if  point[0] < 0 or point[0] >= w  or point[1] < 0 or point[1] >= h :
            vis.append(0)
        else:
            vis.append(1)
    all_keypoints.append({'points' : new_keypoint, 'visibility' : vis})
    return all_keypoints

background_root = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/crawl_box_shutter_stock_high_reso' 
foreground_root = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/Foreground_Extract'
foreground_root_same_border = '/home/tl/ShippingLabelGenerate/crawl_data/download_images/Foreground_same_border'

all_background_img = np.array(get_all_imgpath(background_root))
all_foreground_img = np.array(get_all_imgpath(foreground_root))
all_foreground_same_border_img = np.array(get_all_imgpath(foreground_root_same_border))

write_img_path = 'DocdetectGenerate/img'
write_anno_path = 'DocdetectGenerate/annotation'

num_write = 500
min_size = 800
max_size = 2400
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
eig_vec = np.array([
	[-0.58752847, -0.69563484, 0.41340352],
	[-0.5832747, 0.00994535, -0.81221408],
	[-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

for idx in tqdm(range(num_write*0,num_write*1)) :
    st = time.time()
    same_border = True
    const_var, const_light = np.random.uniform(0,0.2), np.random.uniform(0,0.02)
    if np.random.uniform(0,1) <= 0.2 :
        low, high, mean = 2,8,5
        gen_big_form = True
    else:
        low, high, mean = 1,50,8
        gen_big_form = False
    if np.random.uniform(0,1) <= 0.85 :
        select_foregrounds = all_foreground_img[np.random.randint(0,len(all_foreground_img) - 1, size = rand_int_gaussian(low, high, mean))]
        if np.random.uniform(0,1) <= 0.3 :
            same_border = False
    else:
        select_foregrounds = all_foreground_img[np.random.randint(0,len(all_foreground_same_border_img) - 1, size = rand_int_gaussian(low, high, mean))]
    select_background = all_background_img[np.random.randint(0,len(all_background_img))]
    
	# random_bgr
    back_ground_img = cv2.imread(select_background)
    bgr_h,bgr_w,_ = back_ground_img.shape
    big_size_low = int(max(min_size,max(bgr_h,bgr_w)*0.8))
    big_size_high = max(int(min(max_size,max(bgr_h,bgr_w)*1.5)),big_size_low + 50 )
    big_size = int(np.random.choice(np.arange(big_size_low,big_size_high,30)))
    back_ground_img = cv2.resize(back_ground_img,(int((big_size*bgr_w)/max(bgr_h,bgr_w)),int((big_size*bgr_h)/max(bgr_h,bgr_w))))
    result_img = back_ground_img.copy()
    bgr_h, bgr_w, _ = back_ground_img.shape
    bgr_big_size = max(bgr_h, bgr_w) 
    num_fgr_obj = len(select_foregrounds)
    numfgr = 0

    masks_anno = []
    black_bgr = np.zeros((bgr_h,bgr_w))
    keypoints_anno = []
    for img_path in select_foregrounds:
        # random_fgr
        fgr_img = cv2.imread(img_path)
        fgr_h, fgr_w, _ = fgr_img.shape
        if not gen_big_form :
            max_big_size = bgr_big_size // ( 1 + num_fgr_obj// 5) - 10
            min_big_size = bgr_big_size // (num_fgr_obj * 1.8)
        else:
            max_big_size = int(bgr_big_size / 1.1)
            min_big_size = bgr_big_size // 4

        fgr_big_size = rand_int_gaussian(min_big_size,max_big_size,(3 * max_big_size + 2 * min_big_size)//5)
      
        fgr_w, fgr_h = (int((fgr_big_size*fgr_w)/max(fgr_h,fgr_w)),int((fgr_big_size*fgr_h)/max(fgr_h,fgr_w)))
        if not gen_big_form :
            if np.random.uniform() > 0.8 :
                fgr_h, fgr_w = fgr_w, fgr_h
        else:
            if np.random.uniform() > 0.3 :
                fgr_h, fgr_w = min(fgr_w, fgr_h), max(fgr_w, fgr_h)


        if fgr_h >= bgr_h or fgr_w >= bgr_w :
            fgr_big_size /= 1.3

        fgr_img = cv2.resize(fgr_img,(int((fgr_big_size*fgr_w)/max(fgr_h,fgr_w)),int((fgr_big_size*fgr_h)/max(fgr_h,fgr_w))), interpolation = cv2.INTER_AREA)
        fgr_im_h, fgr_im_w, _ = fgr_img.shape
        bgr = np.zeros((fgr_im_h * 3, fgr_im_w * 3, 3))
        bgr[fgr_im_h : 2*fgr_im_h, fgr_im_w : 2*fgr_im_w] = fgr_img
        fgr_img = bgr
        mask = np.zeros((fgr_im_h * 3, fgr_im_w * 3))
        if np.random.uniform() <= 0.2:
            st_border = 0
        else:
            st_border = 5
        mask[fgr_im_h + st_border : 2*fgr_im_h - st_border, fgr_im_w + st_border : 2*fgr_im_w - st_border ] = 1
        keypoints = [[fgr_im_w + st_border, fgr_im_h + st_border],[2*fgr_im_w - st_border, fgr_im_h + st_border],
                      [fgr_im_w + st_border, 2*fgr_im_h -st_border],[2*fgr_im_w -st_border, 2*fgr_im_h - st_border]]
        keypoints = np.array(keypoints, dtype=np.float32)
        if gen_big_form:
            if np.random.uniform(0,1) <= 0.5:
                degree = (-7,7)
            elif np.random.uniform(0,1) <= 0.7:
                degree = (-15,15)
            elif np.random.uniform(0,1) <= 0.8:
                degree = (-30,30)
            elif np.random.uniform(0,1) <= 0.9:
                degree = (-60,60)
            else:
                degree = (-180,180)
        else:
            degree = (-180,180)
        if same_border :
            fgr_img, mask, keypoints = foreground_random_transform(fgr_img, data_rng, eig_val, eig_vec, mask, degree = degree, keypoint=keypoints)
        else:
            fgr_img, mask, keypoints = foreground_random_transform(fgr_img, data_rng, eig_val, eig_vec, mask,degree = degree, keypoint=keypoints,
                                                                  const_var = const_var, const_light = const_light)
                                                               
        i, j = np.where(mask)
        if len(i) == 0 or len(j) == 0 :
            continue
        min_i, max_i, min_j, max_j = min(i), max(i), min(j), max(j)
        indices = np.meshgrid(np.arange(min_i, max_i + 1), np.arange(min_j, max_j + 1), indexing='ij')
        fgr_img = fgr_img[tuple(indices)]
        mask = mask[tuple(indices)]
        keypoints = keypoints - np.array([min_j,min_i])

        fgr_h, fgr_w, _ = fgr_img.shape
        if (fgr_w >= bgr_w or fgr_h >= bgr_h)  and len(select_foregrounds) <= 4 :
            resize_factor = max(fgr_w / bgr_w, fgr_h / bgr_h)
            rand_factor = np.random.randint(11,13) / 10
            fgr_w /= resize_factor * rand_factor
            fgr_h /= resize_factor * rand_factor
            fgr_img = cv2.resize(fgr_img,(int(fgr_w),int(fgr_h)), interpolation = cv2.INTER_AREA)
            mask  = cv2.resize(mask, (int(fgr_w),int(fgr_h)), interpolation= cv2.INTER_NEAREST)
            keypoints /= float(resize_factor * rand_factor)
        
        fgr_h, fgr_w, _ = fgr_img.shape
        if fgr_w >= bgr_w or fgr_h >= bgr_h :
            continue

        # paste fgr to bgr
        if random.random() > 0.85 :
            display_tl_x = int(np.random.choice(np.arange(-0.7 * fgr_w, bgr_w - 0.3 * fgr_w)))
            display_tl_y = int(np.random.choice(np.arange(-0.7 * fgr_h, bgr_h - 0.3 * fgr_h)))
        else:
            display_tl_x = int(np.random.choice(np.arange(0,bgr_w - fgr_w)))
            display_tl_y = int(np.random.choice(np.arange(0,bgr_h - fgr_h)))
        x1,x2 = max(display_tl_x,0), min(display_tl_x + fgr_w, bgr_w)
        y1,y2 = max(display_tl_y,0), min(display_tl_y + fgr_h, bgr_h)

        fgr_x1, fgr_x2, fgr_y1, fgr_y2 = max( -display_tl_x, 0 ), min(fgr_w, - display_tl_x + bgr_w),\
                                         max( -display_tl_y, 0), min(fgr_h,  - display_tl_y + bgr_h)
        fgr_img =  fgr_img[fgr_y1 : fgr_y2, fgr_x1 : fgr_x2, :]
        mask = mask[fgr_y1 : fgr_y2, fgr_x1 : fgr_x2]
        keypoints = keypoints + np.array([display_tl_x,display_tl_y])
        result_img[y1:y2,x1:x2,:] = copy_paste(result_img[y1:y2,x1:x2,:],fgr_img,mask)         

        black_bgr[y1:y2,x1:x2] = mask
        if len(masks_anno) == 0: 
            masks_anno.append(np.array(black_bgr).astype(np.uint8))
        else :
            masks_anno = update_mask(masks_anno, np.array(black_bgr).astype(np.uint8))        
        keypoints_anno = update_keypoint(keypoints_anno, keypoints, black_bgr)
        black_bgr[y1:y2,x1:x2] = 0 

    bbox_annos = []
    for kp in keypoints_anno:
        points = kp['points']
        min_x, min_y, max_x, max_y = min(points[:,0]),min(points[:,1]), max(points[:,0]), max(points[:,1])
        bbox_annos.append([(min_x, min_y), (max_x, max_y)])

    if len(bbox_annos) > 0 :
        min_box_h = min([box[1][1] - box[0][1] for box in bbox_annos])
        min_box_w = min([box[1][0] - box[0][0] for box in bbox_annos])
        max_cross_h = min(int(min_box_h * 1.2), bgr_h // 5)
        max_cross_w = min(int(min_box_w * 1.2), bgr_w // 5)
        if np.random.uniform() <= 0.3 :
            num_cross = rand_int_gaussian(1,5,2)
            for _ in range(num_cross) :
                rand_color = np.array((np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255)),dtype = np.uint8)
                if np.random.uniform() <= 0.5 :
                    cross_h = np.random.randint(max_cross_h//2, max_cross_h)
                    st = np.random.randint(bgr_h//4, 3*bgr_h//4)
                    cross_mask = np.ones((cross_h,bgr_w - 2))
                    cross_img = np.stack([cross_mask,cross_mask,cross_mask]).transpose(1,2,0) * rand_color
                    rand_combine = np.random.uniform(0.1,0.7)
                    result_img[st:st+cross_h, 1:bgr_w -1] = (1 - rand_combine) * poisson_copy_paste(result_img[st:st+cross_h, 1:bgr_w-1],
                                                        cross_img, cross_mask) + rand_combine * copy_paste(result_img[st:st+cross_h, 1:bgr_w-1],
                                                                                            cross_img, cross_mask)
                else :
                    cross_w = np.random.randint(max_cross_w//2, max_cross_w)
                    st = np.random.randint(bgr_w//4, 3*bgr_w//4)
                    cross_mask = np.ones((bgr_h - 2 ,cross_w))
                    cross_img = np.stack([cross_mask,cross_mask,cross_mask]).transpose(1,2,0) * rand_color
                    rand_combine = np.random.uniform(0.1,0.7)
                    result_img[1:bgr_h-1, st:st+cross_w] = (1 - rand_combine) * poisson_copy_paste(result_img[1:bgr_h-1, st:st+cross_w],
                                                        cross_img, cross_mask) + rand_combine * copy_paste(result_img[1:bgr_h-1, st:st+cross_w],
                                                        cross_img, cross_mask)


    adjust_mask, adjust_box, adjust_kps = [], [], []   
    for mask,box,kp in zip(masks_anno, bbox_annos, keypoints_anno) :
        box_area = (box[1][0] - box[0][0])*(box[1][1] - box[0][1])
        mask_area_ratio= np.sum(mask) / box_area
        thres_keep_mask = 0.05 if not gen_big_form else 0.08
        if mask_area_ratio >= thres_keep_mask :
            adjust_mask.append(mask)
            adjust_box.append(box)
            adjust_kps.append(kp)
    masks_anno, bbox_annos, keypoints_anno = adjust_mask, adjust_box, adjust_kps


    anno_json = {"images":[{"height": bgr_h, "width": bgr_w, "id": 0, "file_name": str(idx) + '.png'}],
                 "categories": [{"supercategory": "shippping_label", "id": 1, "name": "shippping_label",
                                "keypoints": ["top_left", "top_right","bottom_left","bottom_right"],
                                 "skeleton": [[0,1],[0,2],[1,3],[2,3]]}]}
    anno_json["annotations"] = []
    obj_id = 0
    for mask,box,kp in zip(masks_anno, bbox_annos, keypoints_anno) :
        anno_object = {}
        anno_object["segmentation"] = binary_mask_to_rle(mask)
        anno_object["area"] = np.sum(mask)
        anno_object["iscrowd"] = 1
        anno_object["image_id"] = 0
        x1,y1,x2,y2 = box[0][0],box[0][1],box[1][0], box[1][1]
        anno_object["bbox"] = [x1,y1,x2-x1,y2-y1]
        anno_object["category_id"] = 1
        anno_object["id"] = obj_id
        anno_object["num_keypoints"] = 4
        anno_object['keypoints'] = []
        for point,vis in zip(kp['points'], kp['visibility']) :
            anno_object['keypoints'].extend([point[0], point[1], vis + 1])
        obj_id +=1
        anno_json["annotations"].append(anno_object)


    # for anno in anno_json['annotations']:
    #     mask = pycoco_mask.decode(pycoco_mask.frPyObjects(anno['segmentation'],anno['segmentation'].get('size')[0], anno['segmentation'].get('size')[1]))
    #     np.random.seed()
    #     mask = np.stack([mask,mask,mask]).transpose(1,2,0)
    #     rand_color = np.array((np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255)),dtype = np.uint8)
    #     mask == mask * rand_color
    #     result_img = result_img * (1 - mask) + 0.7 * rand_color * mask + 0.3 * result_img * mask
    # for anno in anno_json['annotations'] :
    #     bbox = anno['bbox']
    #     p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #     rand_color =(np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255))
    #     result_img = cv2.rectangle(result_img.copy(), p1, p2,rand_color, 2)
    # kp_idx_name = { 0 : 'top_left', 1 : 'top_right', 2 : 'bottom_left', 3 : 'bottom_right'}
    # for anno in anno_json['annotations'] :
    #     kp = anno['keypoints']
    #     kp_dict = {'points' : [], 'visibility' : []}
    #     for i in range(len(kp)//3):
    #         kp_dict['points'].append([kp[3*i],kp[3*i+1]])
    #         kp_dict['visibility'].append(kp[3*i+2])
    #     kp = kp_dict
    #     rand_color =(np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255))
    #     for kp_idx, (point,vis) in enumerate(zip(kp['points'], kp['visibility'])) :
    #         if 0 < point[0] and point[0] < bgr_w and 0 < point[1] and point[1] < bgr_h :
    #             result_img = cv2.circle(result_img, center= (int(point[0]), int(point[1])), radius= 5, color = rand_color , thickness= -1)
    #             text = 'vis_' + kp_idx_name[kp_idx] if vis == 2 else 'invis_' + kp_idx_name[kp_idx]
    #             result_img = cv2.putText(result_img, text, (int(point[0]) - 20, int(point[1]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4 ,rand_color, 1)
        
    cv2.imwrite(write_img_path + '/' + str(idx) + '.png', result_img)
    with open(write_anno_path + '/' + str(idx) + '.json', 'w') as f:
        json.dump(anno_json, f, default=myconverter)
