from __future__ import print_function

import argparse
import os
import time

import cv2
import imutils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from imutils.video import FPS, FileVideoStream

from retinaface.config import cfg_detect
from retinaface.data import cfg_mnet, cfg_re50
from retinaface.detect import get_bbox_landms, load_model, cal_priorbox
from retinaface.models.retinaface import RetinaFace

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

resize = 1

# testing begin

vid_path  = 'test.mp4'
cap = FileVideoStream(vid_path).start()
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

    dets = get_bbox_landms(img, net, prior_data, device)



    for b in dets:
        b *= frame.shape[0] / frame_resized.shape[0]
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(frame, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image

    fpstext = "Inference speed: " + \
            str(1/(time.time() - t0))[:5] + " FPS"
    cv2.putText(frame, fpstext,
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 255), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
