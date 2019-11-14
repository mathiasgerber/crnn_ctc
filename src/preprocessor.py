import random

import numpy as np
import cv2


def preprocess(image, image_size, data_augmentation = False):
    """
    Preprocesses the image to fit the 128/32 Image that the model expects. The
    images get normalized too.
    :param image: The given input image.
    :param image_size: the size the image needs to have after preprocessing
    :param data_augmentation: if true, the image gets some random
           transformations
    :return: returns the preprocessed image
    """
    if image is None:
        image = np.zeros([image_size[1], image_size[0]])
    if data_augmentation:
        stretch = (random.random() - 0.5)
        width_stretch = max(int(image.shape[1] * (1 + stretch)), 1)
        image = cv2.resize(image, (width_stretch, image.shape[0]))
    (wtarget, htarget) = image_size
    (h, w) = image.shape
    fx = w / wtarget
    fy = h / htarget
    f = max(fx, fy)
    new_size = (max(min(wtarget, int(w / f)), 1),
                max(min(htarget, int(h / f)), 1))
    image = cv2.resize(image, new_size)
    target = np.ones([htarget, wtarget]) * 255
    target[0:new_size[1], 0:new_size[0]] = image
    image = cv2.transpose(target)
    (m, s) = cv2.meanStdDev(image)
    m = m[0][0]
    s = s[0][0]
    image = image - m
    image = image / s if s > 0 else image
    return image
