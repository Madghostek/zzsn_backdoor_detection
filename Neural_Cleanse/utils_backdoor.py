#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import torch
import h5py
import numpy as np
#import tensorflow as tf
#from keras.preprocessing import image
import cv2
import sys
sys.path.append("..")
from torchvision import models
from torch import nn
# from networks.resnet import ResNet18
# from torchvision.models import resnet18
import open_clip

def dump_image(x, filename, format):
    #img = image.array_to_img(x, scale=False)
    #img.save(filename, format)
    #return
    cv2.imwrite(filename, x)


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            print("h5py keys: ", keys)
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def load_model(model_files, device, num_classes):

    sys.path.append('../')
    sys.path.append('../BadMerging/')
    sys.path.append('../BadMerging/src')

    from BadMerging.src.heads import build_classification_head, get_templates, get_classification_head
    from BadMerging.src.modeling import ClassificationHead, ImageClassifier, ImageEncoder


    model = torch.load(model_files[0],weights_only=False).to(device)
    encoder, train_preprocess, val_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    model.model.transformer = encoder.transformer.to(device)

    classification_head = torch.load(model_files[1], weights_only=False).to(device)
    image_encoder = model
    model = ImageClassifier(image_encoder, classification_head)
    return model
