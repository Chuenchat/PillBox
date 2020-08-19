# @title lib_dataset_pills
import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from lib_dataset_pills_eval import pills_eval
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import pandas as pd
import random

from lib_utils_config_parse import cfg


Circles_CLASSES = ('__background__', 'Circle')

min_count = 3
max_count = 9
min_size = .05
max_size = .15
image_h = 300
image_w = 300
augmented_dir = "drive/My Drive/Datasets/VOCdevkit/VOC2007/JPEGImages"
augmented_files = [f for f in os.listdir(augmented_dir)]

def rand_color():
    return [random.randint(0,255) for i in range(3)]

class CircleDetection(data.Dataset):

    def __init__(self, data_dir=None, image_set=None, preproc=None,
                 target_transform=None, dataset_name='EasyEasy'):
        
        # set paramteres for use
        self.image_set = image_set
        self.preproc = preproc
        self.target_transform = target_transform
        
        # create the sample set for being called
        self.samples = [0] * 800

    def __getitem__(self, index):
        
        # load random background from VOC dataset
        augmented_path = os.path.join(augmented_dir, random.choice(augmented_files))
        augmented = cv2.imread(augmented_path, cv2.IMREAD_COLOR)
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        augmented = cv2.resize(augmented, (300,300))

        # random multiple colors and remove duplicates
        colors = list()
        _colors = [rand_color() for i in range(random.randint(min_count, max_count))]
        for c in _colors:
            if c not in colors + [0,0,0]: colors.append(c)

        # draw circles on a blank image
        circles = list()
        color_image = np.zeros(shape=[image_h, image_w, 3], dtype=np.uint8)
        for color in colors:
            x = random.randint(50,250)
            y = random.randint(50,250)
            radius = int(random.uniform(min_size, max_size) * image_h)
            cv2.circle(color_image, (x,y), radius, color, thickness = -1)
            circles.append([x, y, radius])
            
        total_circles = len(circles)

        # target contains 
        # bbox : [x1, y1, x2, y2]
        # class: only 'circle' -> class_no = 1
        # color: in BGR color space
        # iou  : appearance area
        
        # https://stackoverflow.com/questions/39206986/numpy-get-rectangle-area-just-the-size-of-mask?rq=1
        target = list()
        for i in range(total_circles):
          
            # create mask for bbox and iou
            color = colors[i]
            c1 = tuple(color)
            c2 = tuple([c+1 for c in color])
            mask = cv2.inRange(color_image, c1, c2)
            
            # create mask for iou
            x, y, radius = circles[i]
            full = np.zeros(shape=[image_h, image_w], dtype=np.uint8)
            cv2.circle(full, (x,y), radius, 255, thickness = -1)   
            iou = np.sum(mask) / np.sum(full)
            
            # check appearance
            if iou > .2:
                where = np.array(np.where(mask))
                y1, x1 = np.amin(where, axis=1)
                y2, x2 = np.amax(where, axis=1)
                class_no = 1
                target.append([x1, y1, x2, y2, class_no, iou] + color)
                
        # cast to numpy array and normalization
        target = np.asarray(target, dtype=np.float64)
        target /= np.array([1, 1, 1, 1,
                            1, 1, 255, 255, 255])
      
        # overlay color on augmented image (transparent 0%)
        image = color_image
        image[image==0] = augmented[image==0]
      
        # do preprocessing (not required)
        if self.preproc is not None:
            image, target = self.preproc(image, target)
                        
        return image, target

    def __len__(self):
        return len(self.samples)


    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """

        self._write_voc_results_file(all_boxes)
        aps, map = self._do_python_eval(output_dir)
        return aps, map

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = "drive/My Drive/Datasets/Pills_Datasets/mAP_result"
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(Circles_CLASSES):
            cls_ind = cls_ind 
            if cls == '__background__':
                continue
            # print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    # This line almost kill me
                    # index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        aps = []

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print("a")

        for i, cls in enumerate(Circles_CLASSES):
            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)

            print("b")
            rec, prec, ap = pills_eval(
                                    filename, self.dir_lists, cls, self.image_sets,
                                    ovthresh=0.5,
                                    use_07_metric=use_07_metric)
            aps += [ap]
            # print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # valid_aps = [ap for ap in aps if ap != 0]
        # mean_aps = np.mean(valid_aps) if len(valid_aps) > 0 else 0
        
        print("c")
        print(aps)

        mean_aps = np.mean([aps[0], aps[1], aps[2], aps[7], aps[12]])
        print('Mean AP = {:.4f}'.format(mean_aps))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')
        return aps, mean_aps

    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255,0,0), 3)
        cv2.imwrite('./image.jpg', img)