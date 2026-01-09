from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class CLAHE_Transform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 确保图片是 uint8
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # 1. 如果是灰度图，直接做
        if len(image.shape) == 2 or image.shape[2] == 1:
            if len(image.shape) == 3: img_uint8 = img_uint8[:,:,0]
            img_new = self.clahe.apply(img_uint8)
            img_new = img_new[:,:,np.newaxis]
        # 2. 如果是彩色图，转到 LAB 空间对 L 通道做
        else:
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_new = cv2.merge((l_clahe, a, b))
            img_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)

        return {'imidx':imidx, 'image':img_new, 'label':label}


class RescaleT(object):
    def __init__(self, output_size):
        # output_size 应该是 (height, width) 或者 int
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 确定目标尺寸
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            # 传入格式为 (h, w)
            new_h, new_w = self.output_size

        # 使用 order=1 (双线性) 给图片，order=0 (最近邻) 给 Label
        # 关键修正：resize 接收 (h, w)
        img = transform.resize(image, (new_h, new_w), mode='constant', order=1)
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx':imidx, 'image':img, 'label':lbl}

class ToTensorLab(object):
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 1. 处理 Label (归一化到 0-1)
        if np.max(label) > 1:
            label = label / 255.0
        
        # 2. 处理 Image (归一化并标准化)
        # 既然是淡缺陷，建议简单归一化即可，太复杂的 ImageNet 均值方差有时候会破坏微弱特征
        # 这里保留你原本的 ImageNet 标准化，因为加载预训练权重需要它
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        # 归一化到 [0, 1]
        if np.max(image) > 1:
            image = image / 255.0

        if image.shape[2] == 1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        # 处理 Label 维度
        if len(label.shape) == 2:
            tmpLbl = label[:, :, np.newaxis]
        else:
            tmpLbl = label

        # HWC -> CHW
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))

        return {
            'imidx': torch.from_numpy(imidx), 
            'image': torch.from_numpy(tmpImg).float(), 
            'label': torch.from_numpy(tmpLbl).float()
        }

class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读取图片
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        # 读取 Label
        if 0 == len(self.label_name_list):
            label = np.zeros(image.shape[0:2])
        else:
            label_3 = io.imread(self.label_name_list[idx])
            # 处理 Label 维度
            if len(label_3.shape) == 3:
                label = label_3[:, :, 0]
            else:
                label = label_3

        # 确保 image 是 3 通道 (H, W, 3)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4: # 去掉 Alpha 通道
            image = image[:, :, :3]

        # 确保 label 是 2 维 (H, W) 用于 transform 内部处理
        # Transform 之后会被 ToTensorLab 变成 (1, H, W)

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample