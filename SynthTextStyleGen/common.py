import re
import sys
import signal
from contextlib import contextmanager
import numpy as np
from collections import Counter
import cv2
import numpy as np
from itertools import groupby

class Color: #pylint: disable=W0232
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38    

def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))

def warn(msg):
    print (colorize(Color.YELLOW, msg))

def error(msg):
    print (colorize(Color.RED, msg))

# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(colorize(Color.RED, "   *** Timed out!", highlight=True))
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
def rand_int_gaussian(low,high,mean):
    sigma = min(mean-low,high-mean) 
    rand_int = np.random.normal(mean,sigma)
    if rand_int < low:
        rand_int = int(np.random.uniform(low, mean))
    return int(np.clip(rand_int, low,high))

def order_points_old(pts):
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def region_merger(img,seg):
    def min_area_rec(mask):
        contours,hier = cv2.findContours(mask.copy().astype('uint8'),
                                    mode=cv2.RETR_CCOMP,
                                    method=cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)[::-1]
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box
    def b1_in_b2(b1,b2):
        b2_x1,b2_y1,b2_x2,b2_y2 = b2[0][0], b2[0][1], b2[2][0], b2[2][1]
        b1_x1,b1_y1,b1_x2,b1_y2 = b1[0][0], b1[0][1], b1[2][0], b1[2][1]
        if b1_x1 >= b2_x1 and b1_x1 <= b2_x2 and b1_x2 >= b2_x1 and b1_x2 <= b2_x2 \
           and b1_y1 >= b2_y1 and b1_y1 <= b2_y2 and b1_y2 >= b2_y1 and b1_y2 <= b2_y2:
           return True
        return False
        

    count_regions = Counter(seg.reshape(seg.shape[1]*seg.shape[0]))
    area = []
    label=[]
    for cout_reg in count_regions.items():
        area.append(cout_reg[1])
        label.append(cout_reg[0])
    area, label = np.array(area), np.array(label)
    seg_area_h, seg_area_w =  seg.shape
    big_region_idx = np.where(area>seg_area_h*seg_area_w/(6*6)) 
    small_region_idx = np.where(area<seg_area_h*seg_area_w/(10*10))
    big_labels,big_areas,big_rects = label[big_region_idx],area[big_region_idx],[]
    small_labels,small_areas,small_rects = label[small_region_idx], area[small_region_idx],[]

    for b_l in big_labels:
        mask = seg == b_l
        box = min_area_rec(mask)
        box = order_points_old(box)
        big_rects.append(box)
    for s_l in small_labels:
        mask = seg == s_l
        box = min_area_rec(mask)
        box = order_points_old(box)
        small_rects.append(box)
    for small_label,small_rect in zip(small_labels,small_rects):
        small_label = int(small_label)
        is_cont = False
        for big_label,big_rect in zip(big_labels,big_rects):
            b2_x1,b2_y1,b2_x2,b2_y2 = big_rect[0][0], big_rect[0][1], big_rect[2][0], big_rect[2][1]
            if b1_in_b2(small_rect,big_rect)  and (b2_x2 - b2_x1)*(b2_y2 - b2_y1) / (seg_area_w*seg_area_h) <=0.8:
                is_cont = True
                mask = seg == big_label
                xs,ys = np.where(seg==small_label)            
                for x_ , y_ in zip(xs,ys):
                    seg[x_][y_] = big_label
            if is_cont:
                break
    # count_regions = Counter(seg.reshape(seg.shape[1]*seg.shape[0]))
    # label=[]
    # for cout_reg in count_regions.items():
    #     label.append(cout_reg[0])
    # area, label = np.array(area), np.array(label) 
    # for l in label:
    #     l = int(l)
    #     mask = seg == l
    #     xs,ys = np.where(mask) 
    #     color = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
    #     for x_ , y_ in zip(xs,ys):
    #         img[x_][y_] = color
    # cv2.imwrite('new_seg.png',img)

    return seg

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
        


        

            



