import numpy as np
import h5py
import os, sys, traceback
import os.path as osp

from skimage.util import dtype
from synthgen import *
from common import *
import wget, tarfile
from collections import Counter
import json
from pycocotools import mask as pycoco_mask

NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 30 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
BACKGROUND_ROOT = 'depth_segm' 
FOREGROUND_ROOT = '../crawl_data/download_images/Foreground_Extract' 
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
WRITE_IMG_PATH = 'results/img'
WRITE_ANNO_PATH = 'results/annotation'
BGR_MIN_SIZE = 1200
BGR_MAX_SIZE = 1800




def main(viz=False):
  all_img = [os.listdir(BACKGROUND_ROOT + '/input')[13]]
  num_img = len(all_img)
  for im_index,im_file in enumerate(all_img):
    imname = im_file
    try:
      img = cv2.imread(BACKGROUND_ROOT + '/input/' + im_file)
      depth = cv2.imread(BACKGROUND_ROOT + '/output_monodepth/' + im_file)
      depth = depth[:,:,0]
      seg = np.array(np.load(BACKGROUND_ROOT + '/output_semseg/' + im_file.split('.')[0] + '.npy'), dtype=np.float32)
      bgr_h,bgr_w,_ = img.shape
      big_size_low = int(max(BGR_MIN_SIZE,max(bgr_h,bgr_w)*0.8))
      big_size_high = max(int(min(BGR_MAX_SIZE,max(bgr_h,bgr_w)*1.5)),big_size_low + 50 )
      big_size = int(np.random.choice(np.arange(big_size_low,big_size_high,30)))
      img = cv2.resize(img,(int((big_size*bgr_w)/max(bgr_h,bgr_w)),int((big_size*bgr_h)/max(bgr_h,bgr_w))))
      depth = cv2.resize(depth,(int((big_size*bgr_w)/max(bgr_h,bgr_w)),int((big_size*bgr_h)/max(bgr_h,bgr_w))))
      seg = np.array(cv2.resize(seg,(int((big_size*bgr_w)/max(bgr_h,bgr_w)),int((big_size*bgr_h)/max(bgr_h,bgr_w))),interpolation=cv2.INTER_NEAREST), dtype = np.int32 )
      seg = np.array(seg,dtype=np.float32)
      # seg = region_merger(img,seg)
      count_regions = Counter(seg.reshape(seg.shape[1]*seg.shape[0]))
      area = []
      label=[]
      for cout_reg in count_regions.items():
        area.append(cout_reg[1])
        label.append(cout_reg[0])
      area, label = np.array(area), np.array(label)

      # cv2.imwrite('im.png',img)
      # cv2.imwrite('depth.png',depth)
      # regions_color = {}
      # for k in count_regions.keys():
      #   regions_color[k] = np.array([np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)])
      # for i in range(len(img)):
      #   for j in range(len(img[0])):
      #     img[i][j] = img[i][j] * 0.2 + regions_color[seg[i][j]]*0.8
      # cv2.imwrite('seg.png' ,img)
      h,w, _= img.shape
      RV3 = RendererV3(FOREGROUND_ROOT,im_w = w ,im_h = h,max_time=SECS_PER_IMG)

      print('Processing ', im_index, im_file)
      result_img,anno_json = RV3.render_form(img,depth,seg,area,label,
                            ninstance=1)
      anno_json["images"][0]["file_name"] = 'synth_text_style_' + str(im_index) + '.png'

      bgr_h,bgr_w,_ = img.shape
      # for anno in anno_json['annotations']:
      #   mask = pycoco_mask.decode(pycoco_mask.frPyObjects(anno['segmentation'],anno['segmentation'].get('size')[0], anno['segmentation'].get('size')[1]))
      #   np.random.seed()
      #   mask = np.stack([mask,mask,mask]).transpose(1,2,0)
      #   rand_color = np.array((np.random.uniform(0,255), np.random.uniform(0,255), np.random.uniform(0,255)))
      #   mask == mask * rand_color
      #   result_img = result_img * (1 - mask) + 0.7 * rand_color * mask + 0.3 * result_img * mask

      #   bbox = anno['bbox']
      #   p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
      #   rand_color = tuple(list(rand_color))
      #   result_img = cv2.rectangle(result_img.copy(), p1, p2,rand_color, 2)
      #   kp_idx_name = { 0 : 'top_left', 1 : 'top_right', 2 : 'bottom_left', 3 : 'bottom_right'}

      #   kp = anno['keypoints']
      #   kp_dict = {'points' : [], 'visibility' : []}
      #   for i in range(len(kp)//3):
      #       kp_dict['points'].append([kp[3*i],kp[3*i+1]])
      #       kp_dict['visibility'].append(kp[3*i+2])
      #   kp = kp_dict
      #   for kp_idx, (point,vis) in enumerate(zip(kp['points'], kp['visibility'])) :
      #       if 0 < point[0] and point[0] < bgr_w and 0 < point[1] and point[1] < bgr_h :
      #           result_img = cv2.circle(result_img, center= (int(point[0]), int(point[1])), radius= 5, color = rand_color , thickness= -1)
      #           text = 'vis_' + kp_idx_name[kp_idx] if vis == 2 else 'invis_' + kp_idx_name[kp_idx]
      #           result_img = cv2.putText(result_img, text, (int(point[0]) - 20, int(point[1]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4 ,rand_color, 1)


      cv2.imwrite(WRITE_IMG_PATH + '/' + str(im_index) + '.png', result_img)
      with open(WRITE_ANNO_PATH + '/' + str(im_index) + '.json', 'w') as f:
        json.dump(anno_json, f, default=myconverter)

    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)
