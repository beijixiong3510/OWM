# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


def create_dict(file):
    dict = {}
    handle = open(file)
    for line in handle:
        contents = line.split()
        dict[contents[0]] = contents[1:]
    return dict


def default_loader(img):
    return Image.open(img)


class custom_get_set(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path, img_transform=None, loader=default_loader):
        self.img_list = img_path
        self.label_list = txt_path
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = self.loader(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)



