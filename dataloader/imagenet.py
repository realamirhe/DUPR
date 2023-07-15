from tkinter import Image

import numpy as np
import torchvision.transforms.functional as TF

import torch
from datasets import Dataset
from mpmath.identification import transforms


class ImagesDS(Dataset):
    def __init__(self, imgs_list):
        super().__init__()
        self.imgs_list = imgs_list

    def crop_intersection_transform(self, image, index):
        x, y = image.size
        p = np.random.rand()
        if p <= 0.5:
            l1 = 0
            t1 = 0
            h1 = np.random.randint(int(y / 2), int(3 * y / 4))
            w1 = np.random.randint(int(x / 2), int(3 * x / 4))
            l2 = np.random.randint(int(x / 4), int(x / 2))
            t2 = np.random.randint(int(y / 4), int(y / 2))
            h2 = y - t2
            w2 = x - l2
        else:
            l1 = 0
            h1 = np.random.randint(int(y / 2), int(3 * y / 4))
            t1 = y - h1
            w1 = np.random.randint(int(x / 2), int(3 * x / 4))
            l2 = np.random.randint(int(x / 4), int(x / 2))
            t2 = 0
            w2 = x - l2
            h2 = np.random.randint(int(y / 2), int(3 * y / 4))

        tb = max([t1, t2])
        if tb == t1:
            tb1 = 0
            tb2 = t1 - t2
        else:
            tb2 = 0
            tb1 = t2 - t1
        lb = max([l1, l2])
        if lb == l1:
            lb1 = 0
            lb2 = l1 - l2
        else:
            lb2 = 0
            lb1 = l2 - l1
        hb = min([t1 + h1, t2 + h2]) - tb
        wb = min([l1 + w1, l2 + w2]) - lb

        # crop top, left, height, width
        t1 = TF.crop(image, t1, l1, h1, w1)
        t2 = TF.crop(image, t2, l2, h2, w2)
        # B = TF.crop(image, tb, lb, hb, wb)

        trf1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224], antialias=True),
            transforms.ColorJitter()
        ])

        trf2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224], antialias=True),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(p=1)
        ])

        I1 = trf1(t1)
        I1_x_ratio = I1.size(1) / t1.size[0]
        I1_y_ratio = I1.size(2) / t1.size[1]
        tb1 = tb1 * I1_y_ratio
        lb1 = lb1 * I1_x_ratio

        rb1 = lb1 + wb * I1_x_ratio
        bb1 = tb1 + hb * I1_y_ratio

        I2 = trf2(t2)
        I2_x_ratio = I2.size(1) / t2.size[0]
        I2_y_ratio = I2.size(2) / t2.size[1]

        rotated_left = t2.size[1] - lb2 - wb
        rotated_right = t2.size[1] - lb2
        rotated_top = tb2
        rotated_bottom = tb2 + hb

        rotated_left *= I2_x_ratio
        rotated_right *= I2_x_ratio
        rotated_top *= I2_y_ratio
        rotated_bottom *= I2_y_ratio

        lb2, rb2, tb2, bb2 = rotated_left, rotated_right, rotated_top, rotated_bottom

        B1 = torch.Tensor([0, lb1, tb1, rb1, bb1])
        B2 = torch.Tensor([0, lb2, tb2, rb2, bb2])

        return I1, B1, I2, B2

    def __getitem__(self, index):
        image_path = self.imgs_list[index]
        image = Image.open(image_path)
        if image.mode == "L":
            image = image.convert("RGB")
        I1, B1, I2, B2 = self.crop_intersection_transform(image, index)

        return I1, B1, I2, B2

    def __len__(self):
        return len(self.imgs_list)