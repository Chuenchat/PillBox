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


Shape3D_CLASSES = ('__background__', 'Mill')

def bound(low, high, value):
    return max(low, min(high, value))

class Shape3DDetection(data.Dataset):

    def __init__(self, datadirs, image_sets, preproc=None, target_transform=None,
                 dataset_name='Low_Poly_Mill'):

        self.preproc = preproc
        self.target_transform = target_transform
        self.image_sets = image_sets

        self.ids = list()

        data = list()
        columns_list = [
                        'sample_id' ,
                        'image_id'  , 'image_name',
                        'pill_id'   , 'pill_name' ,'pill_shape',
                        'color1_R'  , 'color1_G'  ,'color1_B'  ,
                        'color2_R'  , 'color2_G'  ,'color2_B'  ,
                        'corner1_X' , 'corner1_Y' ,'corner2_X' ,'corner2_Y' ,
                        'corner3_X' , 'corner3_Y' ,'corner4_X' ,'corner4_Y' ,
                        'corner5_X' , 'corner5_Y' ,'corner6_X' ,'corner6_Y' ,
                        'corner7_X' , 'corner7_Y' ,'corner8_X' ,'corner8_Y' ,
                        'bbox_X'    , 'bbox_Y'    ,'bbox_W'    ,'bbox_H'    ,
                        'iou',
                        "path"]
        self.df_label = pd.DataFrame(data = data, columns = columns_list)
        self.dir_lists = []

        self.image_list_list = []
        self.image_folder_list = []

        for i, (dataset_name, number_of_batch, status) in enumerate(image_sets):

            image_folder = os.path.join(datadirs, dataset_name, 'images')
            anno_path = os.path.join(datadirs, dataset_name, 'label.csv')

            self.dir_lists.append(os.path.join(datadirs, dataset_name))

            df = pd.read_csv(anno_path)
            df['path'] = image_folder

            image_list = df["image_name"].unique()
            self.image_folder_list.append(image_folder)
            self.image_list_list.append(image_list)

            self.df_label = self.df_label.append(df, ignore_index=True)

            if number_of_batch == 'all':
                path_list = [os.path.join(image_folder, image_name) for image_name in image_list]
                self.ids.extend(path_list)
            elif number_of_batch.isdigit():
                # in case batch_size = 16
                for b in range(int(number_of_batch)*16):
                    # i index link to image_list for random image
                    self.ids.append("dataset_number_" + str(i))
            else:
                print("error: number_of_batch as '", number_of_batch, "' is not recogonized.")


    def __getitem__(self, index):
      
        def get_image_path(index):
            if self.ids[index].startswith("dataset_number_"):
                i = int(self.ids[index][15:])
                image_folder = self.image_folder_list[i]
                image_list = self.image_list_list[i]
                image_path = os.path.join(image_folder, random.choice(image_list))
            else:
                image_path = self.ids[index]

            return image_path

        # sometime cannot load some images
        try:
            image_path = get_image_path(index)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape
        except:
            image_path = get_image_path(0)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape
        
        head, tail = os.path.split(image_path)                
        img_data = self.df_label.loc[(self.df_label["path"] == head) & \
                                     (self.df_label["image_name"] == tail)
                                     ]
        
        target = []
        e = 0.00001
        for i in range(len(img_data)):

            bx1 = img_data.iloc[i]["bbox_X"]
            by1 = img_data.iloc[i]["bbox_Y"]
            bx2 = img_data.iloc[i]["bbox_X"] + img_data.iloc[i]["bbox_W"]
            by2 = img_data.iloc[i]["bbox_Y"] + img_data.iloc[i]["bbox_H"]

            bx1, bx2 = [bound(e, width - e, v) for v in [bx1, bx2]]
            by1, by2 = [bound(e, height - e, v) for v in [by1, by2]]

            c = Shape3D_CLASSES.index(img_data.iloc[i]["pill_shape"])

            iou = img_data.iloc[i]["iou"]

            cx1 = img_data.iloc[i]["corner1_X"]
            cy1 = img_data.iloc[i]["corner1_Y"]
            cx2 = img_data.iloc[i]["corner2_X"]
            cy2 = img_data.iloc[i]["corner2_Y"]
            cx3 = img_data.iloc[i]["corner3_X"]
            cy3 = img_data.iloc[i]["corner3_Y"]
            cx4 = img_data.iloc[i]["corner4_X"]
            cy4 = img_data.iloc[i]["corner4_Y"]
            cx5 = img_data.iloc[i]["corner5_X"]
            cy5 = img_data.iloc[i]["corner5_Y"]
            cx6 = img_data.iloc[i]["corner6_X"]
            cy6 = img_data.iloc[i]["corner6_Y"]
            cx7 = img_data.iloc[i]["corner7_X"]
            cy7 = img_data.iloc[i]["corner7_Y"]
            cx8 = img_data.iloc[i]["corner8_X"]
            cy8 = img_data.iloc[i]["corner8_Y"]

            # check to filter extreme case
            cs = [cx1,cy1,cx2,cy2,cx3,cy3,cx4,cy4,cx5,cy5,cx6,cy6,cx7,cy7,cx8,cy8]
            extreme = any([not -1 < c/height < 2 for c in cs])

            if iou > .1 and not extreme:
                target.append([bx1, by1, bx2, by2, c, iou, 
                              cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, 
                              cx5, cy5, cx6, cy6, cx7, cy7, cx8, cy8])
            else:
                target.append([0, 0, 0, 0, 0, 0, 
                               0, 0, 0, 0, 0, 0, 0, 0, 
                               0, 0, 0, 0, 0, 0, 0, 0])
                
        
        target = np.asarray(target, dtype=np.float64)
          
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
        
        
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

        for i, cls in enumerate(Circles_CLASSES):
            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)

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