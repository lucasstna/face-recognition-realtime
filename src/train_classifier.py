import json
import math
import os
import pickle

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from feature_extraction import extract_feature
from models.torchmodel.model_irse import IR_50
from utils import *
import argparse

parser = argparse.ArgumentParser(
    description='facerecog')
parser.add_argument('--dataset', default="data/processed_data/train", type=str,
                    help='Dataset path to train svm')
parser.add_argument('--classifier_path', default='models/svm/face_classifier_torch_kpop.pkl', type=str,
                    help='path to svm classifier')
args = parser.parse_args()


INPUT_IMAGE_SIZE = 112

# Face Embedding

trained_model = IR_50((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
trained_model.load_state_dict(torch.load(
    'models/torchmodel/backbone_ir50_asia.pth'))
trained_model.eval()

if torch.cuda.is_available():
    trained_model.cuda()
    torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def resnet50(images):
    images = torch.from_numpy(images).float().to(device)
    images = images.permute(0, 3, 1, 2)
    images = Variable(images)
    features = l2_norm(trained_model(images.to(device)).cpu())
    return features.detach().numpy()


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def train_classifier():
    try:
        dataset = get_dataset(args.dataset)
        paths, labels = get_image_paths_and_labels(dataset)
        paths, labels = shuffle(paths, labels)
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
            emb_array[start_index:end_index, :] = resnet50(images)

        classifier_filename_exp = os.path.expanduser(args.classifier_path)

        # Train classifier
        print('Training classifier')
        svm = SVC(kernel='linear', probability=True)
        svm.fit(emb_array, labels)

        #knn.fit(emb_array, labels)
        #knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((svm, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    except Exception as e:
        print(str(e))

    return 'OK'


def test_recog():
    with open(args.classifier_path, 'rb') as file:
        svm_model, class_names = pickle.load(file)
    dataset = get_dataset('data/processed_data/validation')
    paths, labels = get_image_paths_and_labels(dataset)
    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        img = img[...,::-1]
        emb_array = extract_feature(img, trained_model)
        predictions = svm_model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)[0]
        best_name = class_names[best_class_indices]
        yhat = class_names[labels[i]]
        print(best_name, yhat)

if __name__ == '__main__':
    #train_classifier()
    test_recog()
