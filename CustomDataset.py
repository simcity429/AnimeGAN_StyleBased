import os
import torch
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

#basic transformation for our dataset
BASIC_TRANSFORM = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)])

class TANOCIv2_Dataset(Dataset):
    def __init__(self, img_size, dataset_path, transform):
        #transform only has to contain data augmentating transformation
        super().__init__()
        self.img_size = img_size
        self.img_list = []
        self.transform = transform
        dir_list = [i for i in os.listdir(dataset_path) if i != 'list.txt'] 
        for dir_name in dir_list:
            base_path = os.path.join(dataset_path, dir_name)
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
            img = TF.normalize(img, [0.5,0.5,0.5], [0.5,0.5,0.5])
            return img
        else:
            img = TF.resize(img, self.img_size)
            img = self.transform(img)
            img = TF.center_crop(img, self.img_size)
            img = TF.to_tensor(img)
            img = TF.normalize(img, [0.5,0.5,0.5], [0.5,0.5,0.5])
            return img

