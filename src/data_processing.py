from __future__ import print_function

import argparse
import os
import time

import cv2
import imutils
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from retinaface.config import cfg_detect
from retinaface.data import cfg_mnet, cfg_re50
from retinaface.detect import get_bbox_landms, load_model, cal_priorbox
from retinaface.models.retinaface import RetinaFace

from src.utils import *


torch.set_grad_enabled(False)
cfg = None
if cfg_detect['network'] == "mobile0.25":
    cfg = cfg_mnet
elif cfg_detect['network'] == "resnet50":
    cfg = cfg_re50
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, cfg_detect['trained_model'], cfg_detect['cpu'])
net.eval()
print('Finished loading model!')
cudnn.benchmark = True
device = torch.device("cpu" if cfg_detect['cpu'] else "cuda")
net = net.to(device)

resize = 1

# testing begin



def process_data(raw_dir='./data/raw', processed_dir='./data/processed'):
    try:
        os.mkdir(processed_dir)
    except:
        print(processed_dir, "exist")
    img_paths = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            #append the file name to the list
            img_paths.append(os.path.join(root,file))

    for img_path in img_paths:
        img_name = img_path.split(os.sep)[-1]
        img_id = img_path.split(os.sep)[-2]
        processed_id = os.path.join(processed_dir, img_id)
        out_path = os.path.join(processed_id, img_name)
        if os.path.isfile(os.path.join(processed_id, img_name)):
            print("Image was processed: ", img_path)
            #if image was processed, skip 
            continue
        if not os.path.isdir(processed_id):
            os.mkdir(processed_id)
        
        img = cv2.imread(img_path)
        img_float32 = np.float32(img)

        prior_data = cal_priorbox(cfg, img.shape[:2], device)
        bbox = get_bbox_landms(img_float32, net, prior_data, device)
        if bbox is None:
            continue
        bbox = bbox[0]
        bbox = list(map(int, bbox))
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if cv2.imwrite(out_path, face):
            print("Successful processing: ", img_path)
