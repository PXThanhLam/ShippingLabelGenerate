from __future__ import division
import copy
import cv2
import h5py
from PIL import Image
import numpy as np 
#import mayavi.mlab as mym
import matplotlib.pyplot as plt 
import os.path as osp
import scipy.ndimage as sim
import scipy.spatial.distance as ssd
from skimage.util import dtype
import synth_utils as su
from colorize3_poisson import Colorize
from common import *
import traceback, itertools
import form_utils as fu
from skimage.filters import gaussian



class FormRegions(object):
    minW = 20
    def __init__(self,height,width) -> None:
        self.minWidth = width/15 #px
        self.minHeight = height/15 #px
        self.minAspect = 0.2 # w > 0.2*h
        self.maxAspect = 7
        self.minArea = self.minWidth*self.minHeight
        self.pArea = 0.30 # area_obj/area_minrect >= 0.6

        # RANSAC planar fitting params:
        self.dist_thresh = 0.3 # m
        self.num_inlier = 90
        self.ransac_fit_trials = 100
        self.min_z_projection = 0.2

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"
        """
        wx = np.median(np.sum(mask,axis=0))
        wy = np.median(np.sum(mask,axis=1))
        return wx>FormRegions.minW and wy>FormRegions.minW

    @staticmethod
    def get_hw(pt,return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt,axis=0)
        pt = (pt-mu[None,:]).dot(R.T) + mu[None,:]
        h,w = np.max(pt,axis=0) - np.min(pt,axis=0)
        if return_rot:
            return h,w,R
        return h,w
 
    def filter(self,seg,area,label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > self.minArea]
        area = area[area > self.minArea]
        filt,R = [],[]
        # debug_im =  np.zeros_like(seg)
        # debug_im = np.stack([debug_im,debug_im,debug_im]).transpose(1,2,0)
        for idx,i in enumerate(good):
            mask = seg==i
            xs,ys = np.where(mask) 
            coords = np.c_[xs,ys].astype('float32')
            rect = cv2.minAreaRect(coords)          
            box = np.array(cv2.boxPoints(rect))
            h,w,rot = self.get_hw(box,return_rot=True)

            # print(idx,area[idx],area[idx]/ (w*h) )
            f = (h > self.minHeight 
                and w > self.minWidth
                and self.minAspect < w/h < self.maxAspect
                and area[idx]/ (w*h) > self.pArea)
            # if f :
            #     color = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
            #     for x_ , y_ in zip(xs,ys):
            #         debug_im[x_][y_] = color
            filt.append(f)
            R.append(rot)
        # cv2.imwrite('debug1.png', debug_im)

        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label':good, 'rot':R, 'area': area[aidx]}
        return filter_info

    def sample_grid_neighbours(self,mask,nsample,step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2*step >= min(mask.shape[:2]):
            return #None

        y_m,x_m = np.where(mask)
        mask_idx = np.zeros_like(mask,'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i],x_m[i]] = i

        xp,xn = np.zeros_like(mask), np.zeros_like(mask)
        yp,yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:,:-2*step] = mask[:,2*step:]
        xn[:,2*step:] = mask[:,:-2*step]
        yp[:-2*step,:] = mask[2*step:,:]
        yn[2*step:,:] = mask[:-2*step,:]
        valid = mask&xp&xn&yp&yn

        ys,xs = np.where(valid)
        N = len(ys)
        if N==0: #no valid pixels in mask:
            return #None
        nsample = min(nsample,N)
        idx = np.random.choice(N,nsample,replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs,ys = xs[idx],ys[idx]
        s = step 
        X = np.transpose(np.c_[xs,xs+s,xs+s,xs-s,xs-s][:,:,None],(1,2,0))
        Y = np.transpose(np.c_[ys,ys+s,ys-s,ys+s,ys-s][:,:,None],(1,2,0))
        sample_idx = np.concatenate([Y,X],axis=1)
        mask_nn_idx = np.zeros((5,sample_idx.shape[-1]),'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:,i] = mask_idx[sample_idx[:,:,i][:,0],sample_idx[:,:,i][:,1]]
        return mask_nn_idx

    def filter_depth(self,xyz,seg,regions):
        plane_info = {'label':[],
                      'coeff':[],
                      'support':[],
                      'rot':[],
                      'area':[]}
        # debug_im =  np.zeros_like(seg)
        # debug_im = np.stack([debug_im,debug_im,debug_im]).transpose(1,2,0)
        for idx,l in enumerate(regions['label']):
            mask = seg==l
            pt_sample = self.sample_grid_neighbours(mask,self.ransac_fit_trials,step=7)
            if pt_sample is None:
                continue #not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = su.isplanar(pt, pt_sample,
                                     self.dist_thresh,
                                     self.num_inlier,
                                     self.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2])>self.min_z_projection and np.abs(plane_coeff[0]) > 0 and np.abs(plane_coeff[1]) > 0:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

                    # xs,ys = np.where(mask) 
                    # color = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
                    # for x_ , y_ in zip(xs,ys):
                    #     debug_im[x_][y_] = color
        # cv2.imwrite('debug2.png', debug_im)
        return plane_info

    def get_regions(self,xyz,seg,area,label):
        print('Num of origin region :', len(label))
        regions = self.filter(seg,area,label)
        print('Num of filter region 1 :', len(regions['label']))
        # fit plane to form-regions:
        regions = self.filter_depth(xyz,seg,regions)
        print('Num of filter region 2 : ', len(regions['label']))
        return regions

def rescale_frontoparallel(p_fp,box_fp,p_im):
    l1 = np.linalg.norm(box_fp[1,:]-box_fp[0,:])
    l2 = np.linalg.norm(box_fp[1,:]-box_fp[2,:])

    n0 = np.argmin(np.linalg.norm(p_fp-box_fp[0,:][None,:],axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp-box_fp[1,:][None,:],axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp-box_fp[2,:][None,:],axis=1))

    lt1 = np.linalg.norm(p_im[n1,:]-p_im[n0,:])
    lt2 = np.linalg.norm(p_im[n1,:]-p_im[n2,:])

    s =  max(lt1/l1,lt2/l2)
    if not np.isfinite(s):
        s = 1.0
    return s

def get_form_placement_mask(xyz,mask,plane,pad=2,viz=False):
    contour,hier = cv2.findContours(mask.copy().astype('uint8'),
                                    mode=cv2.RETR_CCOMP,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour_idx = sorted(range(len(contour)), key = lambda k: cv2.contourArea(contour[k]))[::-1]
    contour = [contour[i] for i in contour_idx[:1]]
    hier = np.array([[hier[0][i] for i in contour_idx[:1]]])
    contour = [np.squeeze(c).astype('float') for c in contour]
    #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
    H,W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    pts,pts_fp = [],[]
    center = np.array([W,H])/2
    n_front = np.array([0.0,0.0,-1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = su.DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = su.rot3d(plane[:3],n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:,:2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    
    box = np.array(cv2.boxPoints(rect))
    R2d = su.unrotate2d(box.copy())
    if R2d[0][0] < 0:
        R2d[0][0] = -R2d[0][0]
        R2d[1][1] = -R2d[1][1]
    box = np.vstack([box,box[0,:]]) #close the box for visualization

    mu = np.median(pts_fp[0],axis=0)
    pts_tmp = (pts_fp[0]-mu[None,:]).dot(R2d.T) + mu[None,:]
    boxR = (box-mu[None,:]).dot(R2d.T) + mu[None,:]
    
    # rescale the unrotated 2d points to approximately
    # the same scale as the target region:
    s = rescale_frontoparallel(pts_tmp,boxR,pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s*((pts_fp[i]-mu[None,:]).dot(R2d.T) + mu[None,:])

    # paint the unrotated contour points:
    minxy = -np.min(boxR,axis=0) + pad//2
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:,0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:,1]).T))

    place_mask = 255*np.ones((int(np.ceil(COL))+pad, int(np.ceil(ROW))+pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i]+minxy[None,:]).astype('int32') for i in range(len(pts_fp))]
    cv2.drawContours(place_mask,pts_fp_i32,-1,0,
                     thickness=cv2.FILLED,
                     lineType=8,hierarchy=hier)
    
    if not FormRegions.filter_rectified((~place_mask).astype('float')/255):
        return

    # calculate the homography
    H,_ = cv2.findHomography(pts[0].astype('float32').copy(),
                             pts_fp_i32[0].astype('float32').copy(),
                             method=0)

    Hinv,_ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                pts[0].astype('float32').copy(),
                                method=0)
    return place_mask,H,Hinv

def viz_masks(fignum,rgb,seg,depth,label):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.
    """
    def mean_seg(rgb,seg,label):
        mim = np.zeros_like(rgb)
        for i in np.unique(seg.flat):
            mask = seg==i
            col = np.mean(rgb[mask,:],axis=0)
            mim[mask,:] = col[None,None,:]
        mim[seg==0,:] = 0
        return mim

    mim = mean_seg(rgb,seg,label)

    img = rgb.copy()
    for i,idx in enumerate(label):
        mask = seg==idx
        rgb_rand = (255*np.random.rand(3)).astype('uint8')
        img[mask] = rgb_rand[None,None,:] 

   

    plt.close(fignum)
    plt.figure(fignum)
    ims = [rgb,mim,depth,img]
    for i in range(len(ims)):
        plt.subplot(2,2,i+1)
        plt.imshow(ims[i])
    plt.show(block=False)

def viz_regions(img,xyz,seg,planes,labels):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.
    """
    # plot the RGB-D point-cloud:
    su.plot_xyzrgb(xyz.reshape(-1,3),img.reshape(-1,3))

    # plot the RANSAC-planes at the text-regions:
    for i,l in enumerate(labels):
        mask = seg==l
        xyz_region = xyz[mask,:]
        su.visualize_plane(xyz_region,np.array(planes[i]))


 
def viz_textbb(fignum,text_im, bb_list,alpha=1.0):
    """
    text_im : image containing text
    bb_list : list of 2x4xn_i boundinb-box matrices
    """
    plt.close(fignum)
    plt.figure(fignum)
    plt.imshow(text_im)
    H,W = text_im.shape[:2]
    for i in range(len(bb_list)):
        bbs = bb_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

class RendererV3(object):

    def __init__(self, data_dir, im_w,im_h,max_time=None):
        self.form_renderer = fu.RenderForm(data_dir)
        self.min_char_height = 8 #px
        self.min_asp_ratio = 0.4 #

        self.max_form_regions = 20
        self.max_num_form = 50
        self.max_time = max_time
        self.form_region = FormRegions(height=im_h,width=im_w)
        

    def filter_regions(self,regions,filt):
        """
        filt : boolean list of regions to keep.
        """
        idx = np.arange(len(filt))[filt]
        for k in regions.keys():
            regions[k] = [regions[k][i] for i in idx]
        return regions

    def filter_for_placement(self,xyz,seg,regions):
        # debug_im =  np.zeros_like(seg)
        # debug_im = np.stack([debug_im,debug_im,debug_im]).transpose(1,2,0)
        filt = np.zeros(len(regions['label'])).astype('bool')
        masks,Hs,Hinvs = [],[], []
        for idx,l in enumerate(regions['label']):
            res = get_form_placement_mask(xyz,seg==l,regions['coeff'][idx],pad=2)
            if res is not None:
                mask,H,Hinv = res
                masks.append(mask)
                Hs.append(H)
                Hinvs.append(Hinv)
                filt[idx] = True

                # xs,ys =   np.where(seg == l) 
                # color = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
                # for x_ , y_ in zip(xs,ys):
                #     debug_im[x_][y_] = color
        regions = self.filter_regions(regions,filt)
        regions['place_mask'] = masks
        regions['homography'] = Hs
        regions['homography_inv'] = Hinvs

        # cv2.imwrite('debug3.png', debug_im)
        return regions

    def warpHomography(self,src_mat,H,dst_size):
        dst_mat = cv2.warpPerspective(src_mat, H, dst_size,
                                      flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        return dst_mat

    def homographyBB(self, bbs, H, offset=None):
        """
        Apply homography transform to bounding-boxes.
        BBS: 2 x 4 x n matrix  (2 coordinates, 4 points, n bbs).
        Returns the transformed 2x4xn bb-array.

        offset : a 2-tuple (dx,dy), added to points before transfomation.
        """
        eps = 1e-16
        # check the shape of the BB array:
        t,f,n = bbs.shape
        assert (t==2) and (f==4)

        # append 1 for homogenous coordinates:
        bbs_h = np.reshape(np.r_[bbs, np.ones((1,4,n))], (3,4*n), order='F')
        if offset != None:
            bbs_h[:2,:] += np.array(offset)[:,None]

        # perpective:
        bbs_h = H.dot(bbs_h)
        bbs_h /= (bbs_h[2,:]+eps)

        bbs_h = np.reshape(bbs_h, (3,4,n), order='F')
        return bbs_h[:2,:,:]

    def place_form(self,rgb,collision_mask,H,Hinv,num_blend_form):
        form = self.form_renderer.form_state.sample()

        render_res = self.form_renderer.render_sample(form,collision_mask,num_blend_form)
        if render_res is None: # rendering not successful
            return #None
        else:
            form_mask,form_img,keypoint = render_res

        # warp the object mask back onto the image:
        form_mask = self.warpHomography(form_mask,H,rgb.shape[:2][::-1])
        form_img = self.warpHomography(form_img,H,rgb.shape[:2][::-1])
        keypoint_transform = np.zeros((2,4,1))
        keypoint_transform[0,:,0] = keypoint[:,0]
        keypoint_transform[1,:,0] = keypoint[:,1]
        keypoint = self.homographyBB(keypoint_transform,Hinv)
      
        keypoint_res = np.zeros((4,2))
        keypoint_res[:,0] = keypoint[0,:,0]
        keypoint_res[:,1] = keypoint[1,:,0]
        
        form_mask_res = form_mask.copy()
        form_mask  = np.stack([form_mask,form_mask,form_mask]).transpose(1,2,0)
        form_mask = np.array(form_mask * 255 , dtype = np.uint8)

        # rgb = np.array(rgb,dtype=np.uint8)
        # form_img = np.array(form_img,dtype=np.uint8)
        # center = (form_img.shape[1]// 2, form_img.shape[0] // 2)
        # im_final = cv2.seamlessClone(form_img, rgb, form_mask, center, cv2.MIXED_CLONE)
        
        form_mask  = gaussian(form_mask, sigma=1.0, preserve_range=True,multichannel=False)
        ratio =  1
        im_final = ratio * (rgb * (1 - form_mask/255) + form_img/255 * form_mask) #+ (1 - ratio)*im_final


        return im_final,form_mask_res,keypoint_res


    def get_num_form_regions(self, nregions):
        #return nregions
        nmax = min(self.max_form_regions, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0,1.0)
        return int(np.ceil(nmax * rnd))
    @staticmethod
    def update_mask(masks, paste_mask):
        masks = [
            np.logical_and(mask, np.logical_xor(mask, paste_mask)).astype(np.uint8) for mask in masks
        ]
        masks.append(paste_mask)
        return masks
    @staticmethod
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
    
    @staticmethod
    def copy_paste(image_patch, paste_img, paste_mask):
        paste_mask = gaussian(paste_mask, sigma=1.0, preserve_range=True)
        paste_mask = np.stack([paste_mask,paste_mask,paste_mask]).transpose(1,2,0)

        return image_patch * (1 - paste_mask) + paste_img * paste_mask
    @staticmethod
    def poisson_copy_paste(image_patch, paste_img, paste_mask):
        center = (image_patch.shape[1]// 2, image_patch.shape[0] // 2)
        paste_mask = np.stack([paste_mask,paste_mask,paste_mask]).transpose(1,2,0)

        paste_mask = np.array(paste_mask * 255 , dtype = np.uint8)
        paste_img = np.array(paste_img, dtype = np.uint8)
        image_patch = np.array(image_patch, dtype = np.uint8)
        return cv2.seamlessClone(paste_img, image_patch, paste_mask, center, cv2.MIXED_CLONE)

    def render_form(self,rgb,depth,seg,area,label,ninstance=1):        
        try:
            # depth -> xyz
            xyz = su.DepthCamera.depth2xyz(depth)
            
            # find text-regions:
            regions = self.form_region.get_regions(xyz,seg,area,label)

            # find the placement mask and homographies:
            regions = self.filter_for_placement(xyz,seg,regions)

            # finally place some text:
            nregions = len(regions['place_mask'])
            if nregions < 1: # no good region to place text on
                return []
        except:
            # failure in pre-text placement
            #import traceback
            traceback.print_exc()
            return []

        res = []
        for i in range(ninstance):
            place_masks = copy.deepcopy(regions['place_mask'])
            idict = {'img':[], 'charBB':None, 'wordBB':None, 'txt':None}
            print('Num pasting region  : ', nregions)
            m = self.get_num_form_regions(nregions)#np.arange(nregions)#min(nregions, 5*ninstance*self.max_form_regions))
            reg_idx = np.arange(min(2*m,nregions))
            np.random.shuffle(reg_idx)
            reg_idx = reg_idx[:m]

            img = rgb.copy()
            all_kps = []
            all_bbs = []
            all_masks = []

            # process regions: 
            total_form_blend = rand_int_gaussian(1, 50, 15)
            num_form_regions = len(reg_idx)
            reg_range = np.arange(num_form_regions)
            for idx in reg_range:
                ireg = reg_idx[idx]
                num_blend_form = rand_int_gaussian(1,7, min(4,max(3,total_form_blend//num_form_regions)))
                print(num_blend_form)
                for _ in range(num_blend_form) :
                    try:
                        if self.max_time is None:
                            form_render_res = self.place_form(img,place_masks[ireg],
                                                            regions['homography'][ireg],
                                                            regions['homography_inv'][ireg],num_blend_form)
                        else:
                            with time_limit(self.max_time):
                                form_render_res = self.place_form(img,place_masks[ireg],
                                                                regions['homography'][ireg],
                                                                regions['homography_inv'][ireg],num_blend_form)
                    except TimeoutException as msg:
                        print (msg)
                        continue
                    except:
                        traceback.print_exc()
                        # some error in placing text on the region
                        continue

                    if form_render_res is not None:
                        placed = True
                        img,form_mask,keypoint = form_render_res
                        # store the result:
                        all_kps = RendererV3.update_keypoint(all_kps,keypoint,form_mask)
                        if len(all_masks) == 0: 
                            all_masks.append(np.array(form_mask).astype(np.uint8))
                        else :
                            all_masks = RendererV3.update_mask(all_masks, np.array(form_mask).astype(np.uint8))        
        # print(len(all_kps))          
        for kp in all_kps:
            points = kp['points']
            min_x, min_y, max_x, max_y = min(points[:,0]),min(points[:,1]), max(points[:,0]), max(points[:,1])
            all_bbs.append([(min_x, min_y), (max_x, max_y)])
        bgr_h, bgr_w, _ = img.shape
        if len(all_bbs) > 0 :
            min_box_h = min([box[1][1] - box[0][1] for box in all_bbs])
            min_box_w = min([box[1][0] - box[0][0] for box in all_bbs])
            max_cross_h = min(int(min_box_h * 1.2), bgr_h // 5)
            max_cross_w = min(int(min_box_w * 1.2), bgr_w // 5)
            if np.random.uniform() <= 0.4 :
                num_cross = rand_int_gaussian(1,6,3)
                for _ in range(num_cross) :
                    rand_color = np.array((np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255)),dtype = np.uint8)
                    if np.random.uniform() <= 0.5 :
                        cross_h = np.random.randint(max_cross_h//2, max_cross_h)
                        st = np.random.randint(bgr_h//4, 3*bgr_h//4)
                        cross_mask = np.ones((cross_h,bgr_w - 2))
                        cross_img = np.stack([cross_mask,cross_mask,cross_mask]).transpose(1,2,0) * rand_color
                        rand_combine = np.random.uniform(0.1,0.7)
                        img[st:st+cross_h, 1:bgr_w -1] = (1 - rand_combine) * RendererV3.poisson_copy_paste(img[st:st+cross_h, 1:bgr_w-1],
                                                            cross_img, cross_mask) + rand_combine * RendererV3.copy_paste(img[st:st+cross_h, 1:bgr_w-1],
                                                                                                cross_img, cross_mask)
                    else :
                        cross_w = np.random.randint(max_cross_w//2, max_cross_w)
                        st = np.random.randint(bgr_w//4, 3*bgr_w//4)
                        cross_mask = np.ones((bgr_h - 2 ,cross_w))
                        cross_img = np.stack([cross_mask,cross_mask,cross_mask]).transpose(1,2,0) * rand_color
                        rand_combine = np.random.uniform(0.1,0.7)
                        img[1:bgr_h-1, st:st+cross_w] = (1 - rand_combine) * RendererV3.poisson_copy_paste(img[1:bgr_h-1, st:st+cross_w],
                                                            cross_img, cross_mask) + rand_combine * RendererV3.copy_paste(img[1:bgr_h-1, st:st+cross_w],
                                                            cross_img, cross_mask) 
        
      
        adjust_mask, adjust_box, adjust_kps = [], [], []   
        for mask,box,kp in zip(all_masks, all_bbs, all_kps) :
            box_area = (box[1][0] - box[0][0])*(box[1][1] - box[0][1])
            mask_area_ratio= np.sum(mask) / box_area
            thres_keep_mask = 0.05
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
                   
        return img, anno_json
