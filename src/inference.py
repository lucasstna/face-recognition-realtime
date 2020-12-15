"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import pickle
from src.train_classifier import resnet50
import sys
import time
import math

import cv2
import imutils
import numpy as np
import torch
from torch.autograd.variable import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imutils.video import FPS, FileVideoStream
from models.arcface.model_irse import IR_50
from retinaface.config import cfg_detect
from retinaface.data import cfg_mnet, cfg_re50
from retinaface.detect import cal_priorbox, get_bbox_landms, load_model
from retinaface.models.retinaface import RetinaFace
from sklearn.svm import SVC

from src.feature_extraction import extract_feature, l2_norm
from src.utils import *


class FaceInference:

    CLASSIFIER_PATH = 'models/svm/face_classifier_torch.pkl'
    EMB_MODEL = 'models/arcface/backbone_ir50_asia.pth'
    DATASET = 'data/processed'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    cfg = None
    if cfg_detect['network'] == "mobile0.25":
        cfg = cfg_mnet
    elif cfg_detect['network'] == "resnet50":
        cfg = cfg_re50

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, cfg_detect['trained_model'], cfg_detect['cpu'])
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if cfg_detect['cpu'] else "cuda")
    net = net.to(device)

    # Face Embedding
    face_emb = IR_50((112, 112))
    face_emb.eval()
    face_emb.load_state_dict(torch.load(EMB_MODEL))
    if torch.cuda.is_available():
        face_emb.cuda()
        torch.cuda.empty_cache()
    # SVM load
    with open(CLASSIFIER_PATH, 'rb') as file:
        svm_model, class_names = pickle.load(file)


    def __init__(self, vid_path):
        self.vid_path = vid_path
    def reload_svm(self):
        with open(self.CLASSIFIER_PATH, 'rb') as file:
            self.svm_model, self.class_names = pickle.load(file)
    def facerecog_inference(self):

        cap = FileVideoStream(self.vid_path).start()
        prior_data = None
        while cap.more():
            t0 = time.time()
            frame = cap.read()
            if frame is None:
                continue

            frame_resized = imutils.resize(frame, width=360)
            img = np.float32(frame_resized)

            if prior_data is None:
                prior_data = cal_priorbox(self.cfg, img.shape[:2], self.device)

            try:
                dets = get_bbox_landms(img, self.net, prior_data, self.device)

                for b in dets:
                    b *= frame.shape[0] / frame_resized.shape[0]
                    if b[4] < 0.98:
                        continue
                    b = list(map(int, b))
                    cropped_face = frame[b[1]:b[3], b[0]:b[2]]

                    emb_array = extract_feature(cropped_face, self.face_emb)
                    predictions = self.svm_model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]

                    if best_class_probabilities > 0.60:
                        name = self.class_names[best_class_indices[0]]
                    else:
                        name = "Unknown"

                    cv2.putText(frame, "ID: " + name, (b[0], b[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    # confidence = "{b[4]}".format(b[4])
                    cv2.putText(frame, str(best_class_probabilities), (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # landms
                    cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
            except:
                continue

            fpstext = "Inference speed: " + \
                str(1/(time.time() - t0))[:5] + " FPS"
            cv2.putText(frame, fpstext,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 255), 2)
            _, frame = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  frame.tobytes() + b'\r\n')

    def add_faceid(self, id):
        out_path = os.path.join("./data/processed", str(id))
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        cap = FileVideoStream(self.vid_path).start()
        prior_data = None
        index = 0
        while cap.more():
            t0 = time.time()
            frame = cap.read()
            if frame is None:
                continue
            h, w, d = frame.shape
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # margin_h = 0.1  # horizontal
            # margin_top = 0.15
            # crop_left = (int(w*margin_h), int(h*margin_top))
            # crop_right = (int(w*(1-margin_h)), int(h*margin_top + w*(1-2*margin_h)))

            # cv2.rectangle(frame, crop_left, crop_right, (0, 255, 0), 5)

            # crop_frame = frame[crop_left[1]:crop_right[1],
            #                    crop_left[0]:crop_right[0]]
            frame_resized = imutils.resize(frame, width=360)
            img = np.float32(frame_resized)

            if prior_data is None:
                prior_data = cal_priorbox(self.cfg, img.shape[:2], self.device)

            try:
                dets = get_bbox_landms(img, self.net, prior_data, self.device)
                b = dets[0] # get only first face
                b *= frame.shape[0] / frame_resized.shape[0]
                score = b[4] / 2
                b = list(map(int, b))
            
                fname = os.path.join(out_path, f'{t0}_{index}.jpg')
                cropped_face = frame[b[1]:b[3], b[0]:b[2]]
                if cv2.imwrite(fname, cropped_face):
                    index += 1

                cv2.putText(frame, "{:.2f}".format(score), (b[0] + 30, b[1] + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                cv2.rectangle(frame, (b[0], b[1]),
                              (b[2], b[3]), (0, 0, 255), 2)
            except:
                pass
            finally:
                ret, frame = cv2.imencode('.jpg', frame)
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                        frame.tobytes() + b'\r\n')
    
    def _resnet50(self, images):
        images = torch.from_numpy(images).float().to(self.device)
        images = images.permute(0, 3, 1, 2)
        images = Variable(images)
        features = l2_norm(self.face_emb(images).cpu())
        return features.detach().numpy()

    def train_classifier(self):
        # try:
            dataset = get_dataset(self.DATASET)
            paths, labels = get_image_paths_and_labels(dataset)

            print('Number of labels: %d' % len(set(labels)))
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            print('Calculating features for images')
            embedding_size = 512
            batch_size = 32
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = load_data(paths_batch)
                emb_array[start_index:end_index, :] = self._resnet50(images)


            # Train classifier
            print('Training classifier')
            svm = SVC(kernel='linear', probability=True)
            svm.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            classifier_filename_exp = os.path.expanduser(self.CLASSIFIER_PATH)
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((svm, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
        # except Exception as e:
        #     print(e)
        #     return False
        # return True
