from django.shortcuts import render
from django.http import HttpResponse

import os
import cv2
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
machine_learning_path = os.path.join(os.getcwd(), 'ml')
sys.path.append(machine_learning_path)

from lib_utils_config_parse import cfg
from lib_utils_config_parse import cfg_from_file
from lib_utils_config_parse import update_cfg
from lib_modeling_model_builder import create_model
from lib_layers_functions_detection import *

# Create your views here.
def index(request):
	images_path = "static/data/train_real_04/images"
	json_files = json.dumps(os.listdir(images_path))
	return render(request, 'index.html', {"json_files": json_files})

def detect(request):

	#### section1: Define shape
	shapes = ['__background__', 'Round', 'Oblong', 'Oval', 'Square', 
	          'Rectangle', 'Diamond', 'Triangle', 'Pentagon', 'Hexagon', 
	          'Hepagon', 'Octagon', 'Capsule']
	n_keys = len(shapes)
	n_classes = 20 + 1

	# to onehot
	def onehot_shape(shape, seed):
	    if shape == 'Oval': shape = 'Oblong'
	    if shape == 'Octagon': shape = 'Rectangle'
	    v = [int(e == shape) for e in shapes[1:]]
	    if seed: random.Random(seed).shuffle(v)
	    return v

	#### section2: Define pill's name
	# make pill name map with id
	lib_path = "static/data/pill_library"
	lib_data = pd.read_excel(os.path.join(lib_path, "data.xlsx"))

	pill_name_of = {}
	for index, row in lib_data.iterrows():
	    pill_name_of[row['id']] = row['name']
	    # print(row['id'], row['name'])

	#### section3: Load model 1
	cfg_path = "ml/cfgs"
	cfg_file = "fssd_lite_mobilenetv1_train_voc.yml"
	cfg_from_file(os.path.join(cfg_path, cfg_file))

	cfg['DATASET']['DATASET'] = 'pills'  
	cfg['MODEL']['NUM_CLASSES'] = n_classes
	update_cfg()

	model, priorbox = create_model(cfg.MODEL)
	priors = Variable(priorbox.forward(), volatile=True)
	detector = Detect(cfg.POST_PROCESS, priors)

	model_path = "ml/fssd_lite_mobilenet_v1_pills_epoch_50.pth"
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval();

	#### section4: Load model 2
	n_data = 22

	class SuperSiamese(nn.Module):

	    def __init__(self):
	        super().__init__()
	        self.nn_a = nn.Sequential(
	            nn.Linear(n_data, n_data, bias=True),
	        )

	    def forward(self, d1, d2):
	        d1 = self.nn_a(d1)
	        d2 = self.nn_a(d2)
	        d2 = d2.permute(0, 2, 1)

	        return torch.matmul(d1, d2)

	net_path = "ml/pill_the_best_2.pt"
	net = SuperSiamese()
	net = torch.nn.DataParallel(net)
	net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
	net.eval();

	#### section5: Make data2 template
	# #@title load pill name library

	# prepare data2 for matching
	data2 = []

	# declare paths
	dataset_path = "static/data/train_real_04"
	lib_path = "static/data/pill_library"

	# get sample data
	label = pd.read_csv(os.path.join(dataset_path, "label.csv"))
	library_ids = label["library_id"].unique()
	library_ids = sorted(list(library_ids))

	# get reference data
	lib_data = pd.read_excel(os.path.join(lib_path, "data.xlsx"))

	# iteration make data
	for index, row in lib_data.iterrows():
	    if not row['id'] in library_ids: continue
	        
	    # load ref image
	    image_name = str(row['path']) if not isinstance(row['path'], str) else row['path']
	    image_name += '.jpg'
	    pill_image = cv2.imread(os.path.join(lib_path, "images", image_name))
	    pill_image = cv2.cvtColor(pill_image, cv2.COLOR_BGR2RGB)
	    
	    # min_img = np.min(pill_image)
	    # max_img = np.max(pill_image) - np.min(pill_image)
	    # pill_image = (pill_image - min_img) / max_img
	    # pill_image = (pill_image * 255).astype(np.uint8)

	    mask_image = cv2.imread(os.path.join(lib_path, "masks" , image_name))
	    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

	    # crop square
	    h, w, c = np.shape(pill_image)
	    s = 650
	    x1 = int(w/2 - s/2)
	    y1 = int(h/2 - s/2)
	    x2 = int(w/2 + s/2)
	    y2 = int(h/2 + s/2)
	    pill_image = pill_image[y1:y2, x1:x2]
	    mask_image = mask_image[y1:y2, x1:x2]

	    # crop region
	    ret, thresh = cv2.threshold(mask_image, 10, 255, 0)
	    contours, _ = cv2.findContours(thresh, 1, 2)
	    cnt = contours[0]
	    x, y, w, h = cv2.boundingRect(cnt)
	    roi_image = pill_image[y:y+h, x:x+w]
	    roi_image = cv2.resize(roi_image, (128, 128))

	    # color        
	    roi7 = cv2.resize(roi_image, (7, 7))
	    hsv7 = cv2.cvtColor(roi7, cv2.COLOR_RGB2HSV)
	    col_r, col_g, col_b = roi7[3, 3] / 255
	    col_h, col_s, col_v = hsv7[3, 3] / 255
	    
	    # length
	    l_max = max(w, h) / s
	    l_min = min(w, h) / s
	    multi = l_min * l_max
	    ratio = l_min / l_max

	    # shape
	    shape = row['shape']
	    s_vec = onehot_shape(shape, 7)
	    
	    # pack data
	    d2 = [col_h, col_s, col_v, 
	          col_r, col_g, col_b,
	          l_max, l_min, multi, ratio]
	    d2 += s_vec
	    data2.append(d2)

	# fill 20
	for _ in range(len(data2), 20):
	    data2.append([0]*22)

	# standardize and normalize
	data2 = torch.tensor(data2).type(torch.FloatTensor)
	norm_list = [3, 4, 5, 6, 7, 8]
	for f in norm_list:
	    data2[:, f] -= data2[:, f].mean(dim=0)
	    data2[:, f] /= data2[:, f].std(dim=0)
	data2 = data2.unsqueeze(0)

	#### section6: non-maximum suppression
	#@title Non Maximum Suppression

	# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	def bb_intersection_over_union(boxA, boxB):
	    # determine the (x, y)-coordinates of the intersection rectangle
	    xA = max(boxA[0], boxB[0])
	    yA = max(boxA[1], boxB[1])
	    xB = min(boxA[2], boxB[2])
	    yB = min(boxA[3], boxB[3])
	  
	    # compute the area of intersection rectangle
	    interArea = max(0, xB - xA) * max(0, yB - yA)
	  
	    # compute the area of both the prediction and ground-truth
	    # rectangles
	    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	  
	    # compute the intersection over union by taking the intersection
	    # area and dividing it by the sum of prediction + ground-truth
	    # areas - the interesection area
	    iou = interArea / float(boxAArea + boxBArea - interArea)
	  
	    # return the intersection over union value
	    return iou

	def nms_(pred_list):
	    while True:
	        need_update = False
	        for i in range(len(pred_list)):
	            x1a = pred_list[i][0]
	            y1a = pred_list[i][1]
	            x2a = x1a + pred_list[i][2]
	            y2a = y1a + pred_list[i][3]
	            boxA = [x1a, y1a, x2a, y2a]
	            scoreA = pred_list[i][-1]
	            for j in range(i+1, len(pred_list)):
	                x1b = pred_list[j][0]
	                y1b = pred_list[j][1]
	                x2b = x1b + pred_list[j][2]
	                y2b = y1b + pred_list[j][3]
	                boxB = [x1b, y1b, x2b, y2b]
	                scoreB = pred_list[j][-1]
	                iou = bb_intersection_over_union(boxA, boxB)
	                if iou > 0.5:
	                    if scoreA > scoreB:
	                        del pred_list[j]
	                    else:
	                        del pred_list[i]
	                    need_update = True
	                if need_update:
	                    break
	            if need_update:
	                break
	        if not need_update:
	            break
	            
	    return pred_list

	#### section7: Run through 

	image_name = request.POST['image_name']
	image_path = os.path.join(dataset_path, 'images', image_name)
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	raw_h, raw_w, _ = np.shape(image)
	image = cv2.resize(image, (300, 300))

	xx = torch.from_numpy(image)
	xx = xx.type(torch.FloatTensor)
	xx = xx.unsqueeze(0)
	xx[:, :, :, 0] = xx[:, :, :, 0] - 104.
	xx[:, :, :, 1] = xx[:, :, :, 1] - 117.
	xx[:, :, :, 2] = xx[:, :, :, 2] - 123
	xx = xx.permute(0, 3, 1, 2)

	# detect shape
	out1 = model(xx)
	detections = detector.forward(out1)

	# tensor -> list
	pred_list = list()
	img_h, img_w, img_c = np.shape(image)
	for c in range(detections.size(1)):
	    j = 0
	    while detections[0,c,j,0] >= 0.3:
	        score = detections[0,c,j].cpu().numpy()[0]
	        x1, y1, x2, y2 = detections[0,c,j].cpu().numpy()[1:5]
	        x, w = x1, x2 - x1
	        y, h = y1, y2 - y1
	        iou = detections[0,c,j].cpu().numpy()[5]
	        pred_list.append([x, y, w, h, c, score, iou])
	        j+=1

	# non maximum suppression
	pred_list = nms_(pred_list)


	# make data1 for matching pill name
	data1 = []
	img_h, img_w, _ = np.shape(image)
	min_img = np.min(image)
	max_img = np.max(image) - np.min(image)
	image = (image - min_img) / max_img
	image_uint8 = (image * 255).astype(np.uint8)

	for pred in pred_list:

	    # get detected data
	    x, y, w, h, c, score, iou = pred
	    x1, y1, x2, y2 = x, y, x+w, y+h

	    # bound
	    x1, x2 = [max(0, min(img_w - 1, int(a * img_w))) for a in [x1, x2]]
	    y1, y2 = [max(0, min(img_h - 1, int(a * img_h))) for a in [y1, y2]]

	    # crop pill region
	    crop = image_uint8[y1:y2, x1:x2]
	    
	    # color
	    crop7 = cv2.resize(crop, (7, 7))
	    crop7hsv = cv2.cvtColor(crop7, cv2.COLOR_RGB2HSV)
	    col_r, col_g, col_b = crop7[3, 3] / 255
	    col_h, col_s, col_v = crop7hsv[3, 3] / 255

	    # size
	    l_max = max(h, w)
	    l_min = min(h, w)
	    multi = l_min * l_max
	    ratio = l_min / l_max

	    # shape
	    shape = shapes[c]
	    s_vec = onehot_shape(shape, 7)

	    # pack data
	    d1 = [col_h, col_s, col_v, 
	          col_r, col_g, col_b,
	          l_max, l_min, multi, ratio]
	    d1 += s_vec
	    data1.append(d1)

	# fill 20
	for _ in range(len(data1), 20):
	    data1.append([0]*22)

	# standardize and normalize
	data1 = torch.tensor(data1).type(torch.FloatTensor)
	norm_list = [3, 4, 5, 6, 7, 8]
	for f in norm_list:
	    data1[:, f] -= data1[:, f].mean(dim=0)
	    data1[:, f] /= data1[:, f].std(dim=0)
	data1 = data1.unsqueeze(0)

	# match features
	out2 = net.forward(data1, data2)[0]

	# reformat
	pred_list2 = []
	for i, pred in enumerate(pred_list):
	    x, y, w, h, c, score, iou = pred
	    pill_id = out2[i].max(0)[1].item()
	    # pill_id = library_ids[out2[i].max(0)[1].item() + 1]
	    x1 = int(x * raw_w)
	    y1 = int(y * raw_h)
	    x2 = int((x + w) * raw_w)
	    y2 = int((y + h) * raw_h)
	    pred_list2.append([x1, y1, x2, y2, pill_id])

	json_stuff = json.dumps({"list_of_jsonstuffs" : pred_list2})
	return HttpResponse(json_stuff, content_type ="application/json")