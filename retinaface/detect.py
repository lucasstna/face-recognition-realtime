from __future__ import print_function

import os
import time

import numpy as np
import torch

from retinaface.config import cfg_detect
from retinaface.data import cfg_mnet, cfg_re50
from retinaface.models.retinaface import RetinaFace
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def cal_priorbox(cfg, image_size, device):
    priorbox = PriorBox(cfg, image_size)
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    return prior_data

def get_bbox_landms(img, net, prior_data, device):
    """
    return: dets type ndarray.
    dets.shape = (face founds, 15)
    dets[:,:3]  -> 4 points of each bboxes
    dets[:, 4]   -> score of each faces
    dets[:,5:]  -> 10 points landmark of each faces
    """
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()


    # ignore low scores
    inds = np.where(scores > cfg_detect['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:cfg_detect['top_k']]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg_detect['nms_threshold'])
    # keep = nms(dets, cfg_detect['nms_threshold'],force_cpu=cfg_detect['cpu'])
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:cfg_detect['keep_top_k'], :]
    landms = landms[:cfg_detect['keep_top_k'], :]

    #dets - combine bboxes, scores and landmarks point
    dets = np.concatenate((dets, landms), axis=1)

    return dets
