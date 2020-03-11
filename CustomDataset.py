import os
import torch
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from config import *



class TANOCIv2_Dataset(Dataset):
    def __init__(self, transform=BASIC_TRANSFORM):
        #transform only has to contain data augmentating transformation
        super().__init__()
        self.img_size = IMG_SIZE
        self.img_list = []
        self.transform = transform
        dir_list = [i for i in os.listdir(DATASET_PATH) if i != 'list.txt'] 
        for dir_name in dir_list:
            base_path = os.path.join(DATASET_PATH, dir_name)
            tmp_img_list = os.listdir(base_path)
            for img_name in tmp_img_list:
                path = os.path.join(base_path, img_name)
                self.img_list.append(path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        if self.transform is None:
            img = TF.resize(img, self.img_size)
            img = TF.center_crop(img, self.img_size)
            img = TF.to_tensor(img)
            return img
        else:
            img = TF.resize(img, self.img_size)
            img = self.transform(img)
            img = TF.center_crop(img, self.img_size)
            img = TF.to_tensor(img)
            return img

