import torch 
import os 
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import models, datasets

#define custom dataset for TrojAI competition

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform,triggered_classes,label_specific=False):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        if 'data.csv' in all_imgs:
            all_imgs.remove('data.csv')
        # remove all imgaes except triggered_classes!!!!!!!!
        if label_specific == True:
            dataset_size = len(all_imgs)
            copy_imgs = all_imgs.copy()
            for i in range(dataset_size):
                tmp_img = copy_imgs[i]
                if int(tmp_img.split('_')[-3]) not in triggered_classes:
                    all_imgs.remove(tmp_img)

            
        self.total_imgs = natsorted(all_imgs)
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        label = int(img_loc.split(".jpeg")[0][-1])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image,self.total_imgs[idx],label

class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, index, target
