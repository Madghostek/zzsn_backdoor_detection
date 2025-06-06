from __future__ import absolute_import, print_function

from pathlib import Path
import random

import sys

sys.path.append(".")
sys.path.append("./BadMerging")
sys.path.append("./BadMerging/src")

DATASET = "/home/tomek/datasets/zzsn_vol2"


from BadMerging.src.heads import build_classification_head, get_templates
from BadMerging.src.modeling import ClassificationHead, ImageClassifier, ImageEncoder
import numpy as np
from scipy.stats import gamma
from scipy.stats import median_abs_deviation as MAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

random.seed()

# Detection parameters
NC = 100 # classes
NI = 150
PI = 0.9
NSTEP = 1 #300
TC = 6
batch_size = 20

image_size = [30, 3, 224, 224]

# Load model

# print("Loading model")
encoder,train_preprocess,val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
# setattr(encoder,'val_preprocess',val_preprocess) # trzeba żeby się nie wysypało...
# setattr(encoder,'train_preprocess',train_preprocess)
# num_classes = 100
# embed_dim = encoder.text_projection.shape[-1] if hasattr(encoder, 'text_projection') else encoder.embed_dim
# dummy_weights = torch.empty(num_classes, embed_dim)

# head = ClassificationHead(
#     normalize=True,
#     weights=dummy_weights
# )
# model = ImageClassifier(encoder,head)

# criterion = nn.CrossEntropyLoss()

#if device == 'cuda':
#    model = torch.nn.DataParallel(model)
#    cudnn.benchmark = True

def lr_scheduler(iter_idx):
    lr = 1e-2
    return lr

print(Path("models").absolute())
for checkpoint in Path("models_vol2").glob("*.pt"):
    print("working on",checkpoint)
    model = torch.load(checkpoint,weights_only=False).to(device)
    temp_model,_,_ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    model.model.transformer = temp_model.transformer.to(device)
    template = get_templates("ImageNet100_ZZSN")
    classification_head = build_classification_head(model.model, "ImageNet100_ZZSN", template, DATASET, device)
    model.eval()

    res = []
    for t in range(NC):
        print("Doing class",t)

        images = torch.rand(image_size).to(device)
        images.requires_grad = True

        last_loss = 1000
        labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
        onehot_label = F.one_hot(labels, num_classes=NC)
        for iter_idx in tqdm(range(NSTEP),total=NSTEP):

            optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
            optimizer.zero_grad()
            outputs = model(torch.clamp(images, min=0, max=1)).to(device)
            # print("embedding:",outputs)
            outputs = classification_head(outputs.to(device))
            # print("Outputs:",outputs)

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

    #np.save('results.npy', np.array(res))
    ind_max = np.argmax(stats)
    r_eval = np.amax(stats)
    r_null = np.delete(stats, ind_max)

    shape, loc, scale = gamma.fit(r_null)
    pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1)
    print(pv)
    print(checkpoint.name)
    if pv > 0.05:
        print('No Attack!')
    else:
        print('There is attack with target class {}'.format(np.argmax(stats)))
