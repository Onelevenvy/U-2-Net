from __future__ import print_function, division
import torch

import numpy as np
from skimage import io
import cv2
from torch.utils.data import Dataset



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
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
        return self._clahe

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        # 获取 CLAHE 对象 (首次调用时创建)
        clahe = self._get_clahe()

        # 确保图片是 uint8
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # 1. 如果是灰度图，直接做
        if len(image.shape) == 2 or image.shape[2] == 1:
            if len(image.shape) == 3:
                img_uint8 = img_uint8[:, :, 0]
            img_new = clahe.apply(img_uint8)
            img_new = img_new[:, :, np.newaxis]
        # 2. 如果是彩色图，转到 LAB 空间对 L 通道做
        else:
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_new = cv2.merge((l_clahe, a, b))
            img_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)

        return {"imidx": imidx, "image": img_new, "label": label}



class RescaleT(object):
    """
    图像缩放 Transform (MMDet/Detectron2 风格)
    
    策略: 短边优先缩放，长边硬限制
    
    Args:
        scale: tuple (max_long, min_short) 或 int
            - tuple: (长边最大值, 短边目标值)
            - int: 等价于 (size, size)，即正方形
        keep_ratio: bool，是否保持宽高比 (默认 True)
    
    缩放逻辑 (keep_ratio=True 时):
        1. 先按短边缩放到 min_short
        2. 如果此时长边超过 max_long，改为按长边缩放到 max_long
        3. 这样保证: 长边 <= max_long (硬限制)
        
    示例:
        RescaleT(512)              # 等价于 (512, 512)，方形区域
        RescaleT((1024, 256))      # 长边最大1024，短边目标256
        RescaleT((1333, 800))      # MMDet VOC 常用配置
    
    对于 2048×480 图片，scale=(1024, 256):
        按短边: 480→256, scale=0.533, 长边=1092 (>1024!)
        长边超了，改为: 2048→1024, scale=0.5
        结果: 1024×240
    """
    
    def __init__(self, scale, keep_ratio=True):
        if isinstance(scale, int):
            self.max_long = scale
            self.min_short = scale
        else:
            self.max_long, self.min_short = scale
        self.keep_ratio = keep_ratio

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]
        h, w = image.shape[:2]
        
        if self.keep_ratio:
            long_side = max(h, w)
            short_side = min(h, w)
            
            # Step 1: 按短边缩放到 min_short
            scale = self.min_short / short_side
            
            # Step 2: 检查长边是否超限
            new_long = long_side * scale
            if new_long > self.max_long:
                # 长边超了，改为按长边缩放
                scale = self.max_long / long_side
            
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            # 直接拉伸到目标尺寸 (方形)
            new_h, new_w = self.min_short, self.min_short
        
        # 图像缩放 (双线性插值)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 标签缩放 (最近邻插值，保证标签值不被改变)
        lbl = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return {"imidx": imidx, "image": img, "label": lbl}


class PadToMultiple(object):
    """
    将图像 padding 到指定倍数
    
    通常用于 U-Net 类网络，确保特征图尺寸能被正确下采样和上采样。
    Padding 在图像右下角添加（与 MMDet 一致）。
    
    Args:
        divisor: int，尺寸需要是该数的倍数 (默认 32)
        pad_val: int/float，图像 padding 填充值 (默认 0)
        seg_pad_val: int，标签 padding 填充值 (默认 0，即背景)
    """
    
    def __init__(self, divisor=32, pad_val=0, seg_pad_val=0):
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]
        h, w = image.shape[:2]
        
        # 计算需要 padding 到的目标尺寸
        target_h = int(np.ceil(h / self.divisor)) * self.divisor
        target_w = int(np.ceil(w / self.divisor)) * self.divisor
        
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h == 0 and pad_w == 0:
            return sample
        
        # Padding 图像 (右下角)
        if len(image.shape) == 3:
            img = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(self.pad_val, self.pad_val, self.pad_val)
            )
        else:
            img = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=self.pad_val
            )
        
        # Padding 标签 (右下角)
        lbl = cv2.copyMakeBorder(
            label, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=self.seg_pad_val
        )
        
        return {"imidx": imidx, "image": img, "label": lbl}


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
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

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
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        # 处理 Label 维度
        if len(tmpLbl.shape) == 2:
            tmpLbl = tmpLbl[:, :, np.newaxis]

        # HWC -> CHW
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))

        return {
            "imidx": torch.from_numpy(imidx),
            "image": torch.from_numpy(tmpImg).float(),
            "label": torch.from_numpy(tmpLbl).float(),
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
        image = cv2.imdecode(
            np.fromfile(self.image_name_list[idx], dtype=np.uint8), cv2.IMREAD_COLOR
        )
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
            label = cv2.imdecode(
                np.fromfile(self.label_name_list[idx], dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
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

        sample = {"imidx": imidx, "image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
