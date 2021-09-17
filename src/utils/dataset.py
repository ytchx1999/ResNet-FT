import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

label_mp = {'ants': 0, 'bees': 1}


# download url
# https://download.pytorch.org/tutorial/hymenoptera_data.zip
class AntDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(AntDataset, self).__init__()
        self.data_dir = data_dir
        self.data_info = self.get_data(self.data_dir)
        self.transform = transform

    def __getitem__(self, item):
        img_path, label = self.data_info[item]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data(data_dir):
        data_info = []
        for root, dirs, files in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(data_dir, sub_dir))
                for img_name in img_names:
                    img_path = os.path.join(data_dir, sub_dir, img_name)
                    label = label_mp[sub_dir]
                    data_info.append((img_path, int(label)))
        return data_info


if __name__ == '__main__':
    train_dataset = AntDataset('../../data/hymenoptera_data/train', transform=None)
    print(train_dataset.data_info)
    val_dataset = AntDataset('../../data/hymenoptera_data/val', transform=None)
    print(val_dataset.data_info)
