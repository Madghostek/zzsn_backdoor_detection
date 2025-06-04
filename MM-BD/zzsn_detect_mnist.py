from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import models
import torchvision.transforms as transforms

import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np
from pathlib import Path

from src.resnet import ResNet18
from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Detection parameters
NC = 10 # classes
NI = 150
PI = 0.9
NSTEP = 300
TC = 6
batch_size = 20

# Load model

class CNN(nn.Module):
    def __init__(self, in_shape, out_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.ReLU()

        # convolution parts
        self.c1 = nn.Conv2d(1,3,kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2) # 14,14
        self.c2 = nn.Conv2d(3,6, kernel_size=3, padding=1) # 7x7
        self.c3 = nn.Conv2d(6,12, kernel_size=3, padding=1) # 3x3

        self.fc1 = nn.Linear(12*3*3,50) #12x3x3
        self.fc2 = nn.Linear(50,10)

        # self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # self.fc1 = nn.Linear(128 * 3 * 3, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.act(self.pool(self.c1(x)))
        x = self.act(self.pool(self.c2(x)))
        x = self.act(self.pool(self.c3(x)))

        x = x.flatten(start_dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x) ## !!!!!!!!!!!!!!!!!! ZADNe KURWA RELU DO CROSS ENTROPY, TEZ SIE NIE UCZY NIC
        #x = torch.nn.functional.sigmoid(x) #!!! ZADNA KURWA SIGMOIDA DO CROSS ENTROPY
        return x
criterion = nn.CrossEntropyLoss()

#if device == 'cuda':
#    model = torch.nn.DataParallel(model)
#    cudnn.benchmark = True

def lr_scheduler(iter_idx):
    lr = 1e-2
    return lr

#for checkpoint in Path("models_controlled").glob("*.pth"):
    # model.load_state_dict(torch.load(checkpoint))

model = CNN((1,28,28),NC)
model.to(device)
model.load_state_dict(torch.load('/home/tomek/gitrepos/ML/CNN_kernels/model.pth',device))
model.eval()

res = []
for t in range(NC):

    images = torch.rand([30, 1, 28, 28]).to(device)
    images.requires_grad = True

    last_loss = 1000
    labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
    onehot_label = F.one_hot(labels, num_classes=NC)
    for iter_idx in range(NSTEP):

        optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
        optimizer.zero_grad()
        outputs = model(torch.clamp(images, min=0, max=1))

        loss = -1 * torch.sum((outputs * onehot_label)) \
            + torch.sum(torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
        loss.backward(retain_graph=True)
        optimizer.step()
        if abs(last_loss - loss.item())/abs(last_loss)< 1e-5:
            break
        last_loss = loss.item()

    res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)\
            - torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
    print(t, res[-1])

stats = res
mad = MAD(stats, scale='normal')
abs_deviation = np.abs(stats - np.median(stats))
score = abs_deviation / mad
print(score)

np.save('results.npy', np.array(res))
ind_max = np.argmax(stats)
r_eval = np.amax(stats)
r_null = np.delete(stats, ind_max)

shape, loc, scale = gamma.fit(r_null)
pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1)
print(pv)
if pv > 0.05:
    print('No Attack!')
else:
    print('There is attack with target class {}'.format(np.argmax(stats)))
