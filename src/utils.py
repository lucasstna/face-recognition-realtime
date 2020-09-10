import numpy as np
import os
import cv2

INPUT_IMAGE_SIZE = 112


def transfrom(name):

    punctuation = [',', '\"', '\'', '?', ':', '!', ';']
    for p in punctuation:
        name = name.replace(p, '-')

    return name


def index_largestbb(result):
    return np.argmax((result[:, 3] - result[:, 1]) * (result[:, 2] - result[:, 0]))


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def load_data(image_paths, image_size=INPUT_IMAGE_SIZE, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        resized = cv2.resize(img, (128, 128))
        # center crop image
        a=int((128-112)/2) # x start
        b=int((128-112)/2+112) # x end
        c=int((128-112)/2) # y start
        d=int((128-112)/2+112) # y end
        ccropped = resized[a:b, c:d] # center crop the image
        ccropped = ccropped[...,::-1] # BGR to RGB
        if do_prewhiten:
            ccropped = prewhiten(ccropped)
        images[i, :, :, :] = ccropped
    return images
