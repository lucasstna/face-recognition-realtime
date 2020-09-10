"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import pickle
import sys
import time

import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imutils.video import FPS, FileVideoStream

from src.feature_extraction import extract_feature
from models.torchmodel.model_irse import IR_50
from retinaface.config import cfg_detect
from retinaface.data import cfg_mnet, cfg_re50
from retinaface.detect import get_bbox_landms, load_model, cal_priorbox
from retinaface.models.retinaface import RetinaFace


parser = argparse.ArgumentParser(
    description='facerecog')
parser.add_argument('--path', default="test.mp4", type=str,
                    help='video dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--classifier_path', default='models/svm/face_classifier_torch.pkl', type=str,
                    help='path to svm classifier')
parser.add_argument('--emb_model', default='models/torchmodel/backbone_ir50_asia.pth', type=str,
                    help='dir pth model')
args = parser.parse_args()

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
print(net)
cudnn.benchmark = True
device = torch.device("cpu" if cfg_detect['cpu'] else "cuda")
net = net.to(device)

# Face Embedding
face_emb = IR_50((112, 112))
face_emb.eval()
face_emb.load_state_dict(torch.load(args.emb_model))
if torch.cuda.is_available():
    face_emb.cuda()
    torch.cuda.empty_cache()
# SVM load
with open(args.classifier_path, 'rb') as file:
    svm_model, class_names = pickle.load(file)
time.sleep(2.0)
print("loaded")

cap = FileVideoStream(args.path).start()
fps = FPS().start()
prior_data = None
while cap.more():
    t0 = time.time()
    frame = cap.read()
    if frame is None:
        continue
    
    frame_resized = imutils.resize(frame, width=360)
    img = np.float32(frame_resized)

    if prior_data is None:
        prior_data = cal_priorbox(cfg, img.shape[:2], device)

    try:
        dets = get_bbox_landms(img, net, prior_data, device)

        for b in dets:
            b *= frame.shape[0] / frame_resized.shape[0]
            b = list(map(int, b))
            cropped_face = frame[b[1]:b[3], b[0]:b[2]]

            emb_array = extract_feature(cropped_face, face_emb)
            predictions = svm_model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[
                np.arange(len(best_class_indices)), best_class_indices]

            if best_class_probabilities > 0.50:
                name = class_names[best_class_indices[0]]
            else:
                name = "Unknown"

            cv2.putText(frame, "ID: " + name, (b[0], b[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            confidence = "{:.4f}".format(b[4])
            cv2.putText(frame, confidence, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # landms
            cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
    except :
        continue
    

    fpstext = "Inference speed: " + \
            str(1/(time.time() - t0))[:5] + " FPS"
    cv2.putText(frame, fpstext,
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 255), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

cv2.destroyAllWindows()
cap.stop()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS: {:.2f}".format(fps.fps()))
