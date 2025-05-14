import time 
import numpy as np
import pandas as pd 

import torch 
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from cnn import *

from PIL import Image

import cv2 

use_cuda = torch.cuda.is_available()
print(use_cuda)
print(torch.__version__)
print(torch.cuda.is_available())

DEVICE = torch.device('cuda' if use_cuda else 'cpu')
# MODEL_DIR = '.'  # model directory
MODEL_DIR = './injection'
MODEL_FILENAME = 'gtsrb_backdoor_cnn.pth'  # model file
Img_path = '/home/sense/cleanse_pytorch/data-2/'

preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
def load_model(model_file, device):
    # use_cuda = torch.cuda.is_available()
    # net = resnet18().to(device)
    # print("In load model")
    net = c6f2().to(device)
    model = torch.load(model_file) #original
    # model = torch.load(model_file,  map_location=torch.device('cuda' if use_cuda else 'cpu')) # suggest 但這樣應該就沒有用到GPU?
    net.load_state_dict(model.state_dict())
    net.eval()
    # print("model done")
    # print(net)
    return net




def main():
    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file, DEVICE)

    transform_valid = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.PILToTensor()
    ])

    attack_suc = 0
    ori_attack_suc = 0
    total = 0

    img = Image.open(Img_path + '0_adv.jpg')
    image = transform_valid(img).float()
    image = image.to(DEVICE)

    # print("size:",image.size()) # size: torch.Size([1, 3, 32, 32])
    outputs = model(image)
    # print("out:", outputs[0])
    out = torch.tensor(outputs[0])
    _, ans = torch.max(out,1)
    
    print("Label: ",ans[0].item())
    # print("id: ", ans[1][0])

    # for i in range(0, 29):

    #     img = Image.open(Img_path + str(i) + '_adv.jpg')
    #     image = transform_valid(img).float()
    #     image = image.to(DEVICE)

    #     # print("size:",image.size()) # size: torch.Size([1, 3, 32, 32])
    #     outputs = model(image)
    #     # print("out:", outputs[0])
    #     out = torch.tensor(outputs[0])
    #     _, ori_ans = torch.max(out,1)
    #     print("id: ", ori_ans)

    #     if ans == ori_ans:
    #         ori_attack_suc += 1
        
    #     for j in range(7, 10):
    #         img = Image.open(Img_path + str(i) + '_test_' + str(j) + '.jpg')
    #         image = transform_valid(img).float()
    #         image = image.to(DEVICE)

    #         # print("size:",image.size()) # size: torch.Size([1, 3, 32, 32])
    #         outputs = model(image)
    #         # print("out:", outputs[0])
    #         out = torch.tensor(outputs[0])
    #         _, indices = torch.max(out,1)
    #         print("id: ", indices)
    #         total += 1

    #         if ans == indices:
    #             attack_suc += 1


    # print('ori_attack_suc: ', ori_attack_suc)
    # print('ori_attack_suc_rate: ', float(ori_attack_suc / 29))
    # print('total: ', total)
    # print('attack_suc: ', attack_suc)    
    # print('attack_suc_rate: ', float(attack_suc / total))
    

    


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
