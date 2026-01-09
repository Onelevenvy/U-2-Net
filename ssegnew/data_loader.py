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
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) Transform.
    
    注意：cv2.CLAHE 对象无法被 pickle 序列化，所以不能在 __init__ 中创建。
    这会导致 Windows 上多进程 DataLoader (num_workers > 0) 报错。
    解决方案：在 __call__ 中延迟创建 CLAHE 对象。
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # 只保存参数，不创建 CLAHE 对象 (避免 pickle 问题)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None  # 延迟初始化

    def _get_clahe(self):
        """惰性创建 CLAHE 对象"""
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return self._clahe

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        # 获取 CLAHE 对象 (首次调用时创建)
        clahe = self._get_clahe()

        # 确保图片是 uint8
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # 1. 如果是灰度图，直接做
        if len(image.shape) == 2 or image.shape[2] == 1:
            if len(image.shape) == 3: img_uint8 = img_uint8[:,:,0]
            img_new = clahe.apply(img_uint8)
            img_new = img_new[:,:,np.newaxis]
        # 2. 如果是彩色图，转到 LAB 空间对 L 通道做
        else:
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
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
    """
    转换为 Tensor 并进行 ImageNet 标准化
    
    Args:
        flag: 保留参数
        num_classes: 类别数
            - 1: 二值分割，标签归一化到 0-1
            - >1: 多类别分割，标签保持类别索引 (0, 1, 2, ...)
    """
    def __init__(self, flag=0, num_classes=1):
        self.flag = flag
        self.num_classes = num_classes

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 1. 处理 Label
        if self.num_classes == 1:
            # 二值分割: 归一化到 0-1
            if np.max(label) > 1:
                label = label / 255.0
            tmpLbl = label.astype(np.float32)
        else:
            # 多类别分割: 保持类别索引 (0, 1, 2, ...)
            # 假设标签已经是类别索引，不需要除以 255
            tmpLbl = label.astype(np.float32)
        
        # 2. 处理 Image (归一化并标准化)
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

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
        if len(tmpLbl.shape) == 2:
            tmpLbl = tmpLbl[:, :, np.newaxis]

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
        # 读取图片 - 使用 cv2 比 skimage.io 快 2-3 倍
        image = cv2.imdecode(np.fromfile(self.image_name_list[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            # 回退到 skimage
            image = io.imread(self.image_name_list[idx])
        else:
            # cv2 读取的是 BGR，需要转成 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        imidx = np.array([idx])

        # 读取 Label
        if 0 == len(self.label_name_list):
            label = np.zeros(image.shape[0:2], dtype=np.uint8)
        else:
            # 使用 cv2 读取灰度图
            label = cv2.imdecode(np.fromfile(self.label_name_list[idx], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if label is None:
                # 回退到 skimage
                label_3 = io.imread(self.label_name_list[idx])
                if len(label_3.shape) == 3:
                    label = label_3[:, :, 0]
                else:
                    label = label_3

        # 确保 image 是 3 通道 (H, W, 3)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # 去掉 Alpha 通道
            image = image[:, :, :3]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample