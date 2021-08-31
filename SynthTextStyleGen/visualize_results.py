# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *
import cv2


def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    
    H,W = text_im.shape[:2]
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        print(bb)
        bb = np.c_[bb,bb[:,0]]
        print('---')
        # plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # # visualize the indiv vertices:
        # vcol = ['r','g','b','k']
        # for j in range(4):
        #     plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        



def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    for k in dsets:
        rgb = np.array(db['data'][k][...], dtype =np.float32)
        cv2.imwrite('z_' + str(k) + '.png',cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # charBB = db['data'][k].attrs['charBB']
        # wordBB = db['data'][k].attrs['wordBB']
        # txt = db['data'][k].attrs['txt']

        # viz_textbb(rgb, [charBB], wordBB)
 

if __name__=='__main__':
    main('results/SynthText.h5')

