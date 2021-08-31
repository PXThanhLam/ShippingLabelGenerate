import cv2
import numpy as np
import random
from random import randint
import math

def rand_int_gaussian(low,high,mean):
    sigma = min(mean-low,high-mean) 
    rand_int = np.random.normal(mean,sigma)
    if rand_int < low:
        rand_int = int(np.random.uniform(low, mean))
    return int(np.clip(rand_int, low,high))
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

def lighting_(data_rng, image, alphastd, eigval, eigvec, const_var = None):
    if const_var is not None :
        alpha = const_var
    else:
        alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var, const_var = None):
    if const_var is not None :
        alpha = 1. + const_var
    else :
        alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var, const_var = None):
    if const_var is not None :
        alpha = 1. + const_var
    else :
        alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var, const_var = None):
    if const_var is not None :
        alpha = 1. + const_var
    else :
        alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image, eig_val, eig_vec, const_var = None, const_light = None):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.2, const_var)
    lighting_(data_rng, image, 0.05, eig_val, eig_vec,const_light)

def gaussian_blurr(image, rand = 0.8) :
    if random.random() > rand :
        return image
    height, width, _ = image.shape
    k_size_h = int(0.001 * height * random.random())
    k_size_w = int(0.001 * width * random.random())
    return cv2.GaussianBlur(image,(2 * k_size_h + 1 ,2 * k_size_w + 1),0)

def random_offset(height, width, shift, mode=0):
    # mode = 0, radom inside image only ( no outlier )
    if mode == 0:
        # top corner left and right
        tl_x, tl_y = 0 + randint(0, shift), 0 + randint(0, shift)
        tr_x, tr_y = width - randint(0, shift), 0 + randint(0, shift)
        # bottom corner left and right
        br_x, br_y = width - randint(0, shift), height - randint(0, shift)
        bl_x, bl_y = 0 + randint(0, shift), height - randint(0, shift)
    else:
        tl_x, tl_y = 0 + randint(-shift, shift), 0 + randint(-shift, shift)
        tr_x, tr_y = width + randint(-shift, shift), 0 + randint(-shift, shift)

        br_x, br_y = width + randint(-shift, shift), height + randint(-shift, shift)
        bl_x, bl_y = 0 + randint(-shift, shift), height + randint(-shift, shift)
    return np.array([tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y])

def warp_image(img, pts_src, pts_dst,keypoint = None):
    try:
        height, width, _ = img.shape
    except ValueError:
        height, width = img.shape
    M, _ = cv2.findHomography(pts_src, pts_dst)
    if keypoint is not None :
        points  = np.float32(keypoint).reshape(-1,1,2)
        keypoint = cv2.perspectiveTransform(points, M)
        keypoint = keypoint[:,0,:]
        keypoint = keypoint[:,:2]

    return cv2.warpPerspective(img, M, dsize=(width, height)), keypoint

def random_perpective(img, mask, shift,  mode, keypoint = None, rand = 0.05) :
    if random.random() > rand :
        return img, mask, keypoint
    height, width, _ = img.shape
    y = random_offset(height, width, shift, mode)

    pts_src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = 'float32')
    pts_dst = np.array(np.reshape(y, (4, 2)), dtype= 'float32')

    warped_image, keypoint  = warp_image(img, pts_src, pts_dst, keypoint = keypoint)
    mask, _ = warp_image(mask, pts_src, pts_dst)
    return warped_image, mask, keypoint

def random_affine(img, mask,degrees=(-180, 180), keypoint = None, shear=(-0.02, 0.02), rand = 0.95) :
    if random.random() > rand :
        return img, mask, keypoint
    height = img.shape[0]
    width = img.shape[1]

    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=1)

    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)
    M = S @ R

    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue = (0,0,0))
    mask = cv2.warpPerspective(mask, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue = (0,0,0))
    if keypoint is not None :
        ones = np.ones(shape=(len(keypoint), 1))
        points_ones = np.hstack([keypoint, ones])
        keypoint = M.dot(points_ones.T).T
        keypoint = keypoint[:,:2]
    return imw, mask, keypoint

def foreground_random_transform(img, data_rng, eig_val, eig_vec, fgr_mask, keypoint, degree = (-180,180),const_var = None, const_light = None,
                               perpective_rand = 0.0, affine_rand = 0.8 ):
    img = np.array(img / 255.0, dtype = np.float32)
    # color_jittering_(data_rng, img, eig_val, eig_vec,const_var, const_light)
    # img = gaussian_blurr(img)
    shift = max(img.shape[:2])//5

    img, fgr_mask, keypoint = random_perpective(img, fgr_mask,  shift = shift, mode = 0, keypoint = keypoint, rand = perpective_rand)
    img, fgr_mask, keypoint = random_affine(img,fgr_mask, degrees = degree, keypoint = keypoint, rand = affine_rand)
    img = np.clip(img * 255,0,255)

    return img, fgr_mask, keypoint
    
if __name__ == '__main__':
    img = cv2.imread('crawl_data/download_images/Foreground_Extract/bill_29.jpeg')
    img = cv2.resize(img,(800,500))
    cv2.imwrite('original.png',img)

    im_h, im_w, _ = img.shape
    bgr = np.zeros((im_h * 2, im_w * 2, 3))
    bgr[im_h//2 : 3*im_h//2, im_w//2 : 3*im_w//2] = img
    img = bgr
    mask = np.zeros((im_h * 2, im_w * 2))
    mask[im_h//2 : 3*im_h//2, im_w//2 : 3*im_w//2] = 1
    keypoint = [[im_w//2, im_h//2],[3*im_w//2, im_h//2],[im_w//2, 3*im_h//2],[3*im_w//2, 3*im_h//2]]
    keypoint = np.array(keypoint, dtype=np.int32)

    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    img, mask, keypoint = foreground_random_transform(img, data_rng, eig_val, eig_vec, mask, keypoint)

    # i, j = np.where(mask)
    # indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
    # img = img[tuple(indices)]
    # mask = mask[tuple(indices)]
    for point in keypoint :
        cv2.circle(img, center= (int(point[0]), int(point[1])), radius= 15, color = (0, 0, 255), thickness= 2)
    cv2.imwrite('transform.png',img)
    cv2.imwrite('mask.png', mask * 255 )