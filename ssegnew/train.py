"""
U2Net 一体化训练脚本
- 自动将 labelme JSON 转换为 U2Net 格式
- 所有数据和模型保存在项目文件夹下
- 只需配置项目名称和源数据路径即可

用法:
    python train.py
"""

import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from datetime import datetime
import time
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
from torchvision import transforms
from loguru import logger

from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import FaintDefectLoss, muti_loss_fusion


cudnn.benchmark = True

# =====================================================================
#                           用户配置区域
# =====================================================================

# 1. 项目名称 (所有数据和模型都保存在 projects/<PROJECT_NAME>/ 下)
PROJECT_NAME = "daowen_b402"

# 2. 源数据路径 (labelme 格式: 图片 + 同名 JSON 文件)
#    支持递归搜索子文件夹
SOURCE_DATA_DIR = r"F:\DL项目测试\4_濠玮b402-刀纹\2_Skolpha\1_train\100pcs"

# 3. 模型配置
MODEL_NAME = "u2netp"  # "u2netp" (轻量版) 或 "u2net" (完整版)

# 4. 训练超参数
INPUT_SIZE = (224, 512)  # (Height, Width)
BATCH_SIZE = 8
EPOCH_NUM = 300
LEARNING_RATE = 1e-3


# 5. 其他配置
NUM_WORKERS = 4          # DataLoader 工作线程数
SAVE_EVERY_N_EPOCHS = 10  # 每隔多少 epoch 保存一次模型

# =====================================================================
#                           自动计算路径
# =====================================================================

# 项目根目录
PROJECT_DIR = os.path.join(os.getcwd(), "projects", PROJECT_NAME)

# 转换后的数据目录
CONVERTED_DATA_DIR = os.path.join(PROJECT_DIR, "train_data")
IMAGES_DIR = os.path.join(CONVERTED_DATA_DIR, "images")
MASKS_DIR = os.path.join(CONVERTED_DATA_DIR, "masks")

# 模型保存目录
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "models")

# TensorBoard 日志目录
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_DIR, "runs")

# 预训练权重路径
PRETRAINED_PATH = os.path.join(os.getcwd(),  "pretrain", f"{MODEL_NAME}.pth")

# =====================================================================
#                           Labelme 转换函数
# =====================================================================

def polygon_to_mask(points, image_width, image_height):
    """将多边形点列表转换为二值mask"""
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    polygon_points = [(int(p[0]), int(p[1])) for p in points]
    draw.polygon(polygon_points, fill=255)
    return np.array(mask)


def convert_labelme_json(json_path, output_mask_path):
    """将单个labelme JSON文件转换为mask图片"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 创建空白mask
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # 遍历所有shapes
        for shape in data.get('shapes', []):
            shape_type = shape.get('shape_type', '')
            points = shape.get('points', [])
            
            if shape_type == 'polygon' and len(points) >= 3:
                mask = polygon_to_mask(points, image_width, image_height)
                combined_mask = np.maximum(combined_mask, mask)
            
            elif shape_type == 'rectangle' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                combined_mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 255
            
            elif shape_type == 'circle' and len(points) == 2:
                cx, cy = int(points[0][0]), int(points[0][1])
                ex, ey = int(points[1][0]), int(points[1][1])
                radius = int(np.sqrt((ex-cx)**2 + (ey-cy)**2))
                cv2.circle(combined_mask, (cx, cy), radius, 255, -1)
        
        # 保存mask
        mask_img = Image.fromarray(combined_mask)
        mask_img.save(output_mask_path)
        return True
        
    except Exception as e:
        logger.error(f"转换失败 {json_path}: {e}")
        return False


def convert_dataset():
    """
    自动将源数据转换为 U2Net 格式
    返回: 是否需要重新转换 (True=已转换, False=失败)
    """
    # 检查是否已经转换过
    if os.path.exists(IMAGES_DIR) and os.path.exists(MASKS_DIR):
        existing_images = len(glob.glob(os.path.join(IMAGES_DIR, "*.*")))
        existing_masks = len(glob.glob(os.path.join(MASKS_DIR, "*.png")))
        
        if existing_images > 0 and existing_masks > 0:
            logger.info(f"检测到已转换的数据: {existing_images} 张图片, {existing_masks} 个mask")
            logger.info("跳过转换步骤，直接使用已有数据")
            return True
    
    logger.info("=" * 50)
    logger.info("开始转换 Labelme 数据...")
    logger.info(f"源数据目录: {SOURCE_DATA_DIR}")
    logger.info(f"输出目录: {CONVERTED_DATA_DIR}")
    logger.info("=" * 50)
    
    # 创建输出目录
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    input_path = Path(SOURCE_DATA_DIR)
    
    # 递归查找所有JSON文件
    json_files = list(input_path.rglob('*.json'))
    
    if len(json_files) == 0:
        logger.error(f"未找到 JSON 文件！请检查源数据目录: {SOURCE_DATA_DIR}")
        return False
    
    logger.info(f"找到 {len(json_files)} 个 JSON 文件")
    
    success_count = 0
    fail_count = 0
    
    for json_file in json_files:
        base_name = json_file.stem
        json_dir = json_file.parent
        
        # 尝试多种图片扩展名
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']:
            possible_image = json_dir / f"{base_name}{ext}"
            if possible_image.exists():
                image_file = possible_image
                break
        
        if image_file is None:
            logger.warning(f"找不到对应的图片: {base_name}")
            fail_count += 1
            continue
        
        # 输出路径
        output_image = Path(IMAGES_DIR) / f"{base_name}{image_file.suffix}"
        output_mask = Path(MASKS_DIR) / f"{base_name}.png"
        
        # 复制图片
        shutil.copy2(image_file, output_image)
        
        # 转换JSON为mask
        if convert_labelme_json(json_file, output_mask):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("=" * 50)
    logger.info(f"转换完成! 成功: {success_count}, 失败: {fail_count}")
    logger.info("=" * 50)
    
    return success_count > 0


# =====================================================================
#                           TensorBoard
# =====================================================================

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


# =====================================================================
#                           训练主函数
# =====================================================================

def train():
    """训练主函数"""
    
    # 1. 准备数据集目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 获取图片和标签文件列表
    image_ext = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        if len(glob.glob(os.path.join(IMAGES_DIR, f"*{ext}"))) > 0:
            image_ext = ext
            break
    
    if image_ext is None:
        logger.error("未找到训练图片！")
        return
    
    tra_img_name_list = glob.glob(os.path.join(IMAGES_DIR, f"*{image_ext}"))
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(MASKS_DIR, f"{base_name}.png")
        tra_lbl_name_list.append(lbl_path)
    
    logger.info(f"训练图片数量: {len(tra_img_name_list)}")
    
    # 2. 定义 DataLoader
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(INPUT_SIZE),
            CLAHE_Transform(), 
            ToTensorLab(flag=0),
        ]),
    )

    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True
    )
    logger.info(f"DataLoader: batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")

    # 3. 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_NAME == "u2net":
        net = U2NET(3, 1)
    elif MODEL_NAME == "u2netp":
        net = U2NETP(3, 1)
    else:
        logger.error(f"未知模型: {MODEL_NAME}")
        return

    net = net.to(device)
    logger.info(f"模型 {MODEL_NAME} 加载到设备: {device}")

    # 4. 加载预训练权重
    if os.path.exists(PRETRAINED_PATH):
        logger.info(f"加载预训练权重: {PRETRAINED_PATH}")
        try:
            net.load_state_dict(torch.load(PRETRAINED_PATH), strict=False)
        except Exception as e:
            logger.warning(f"预训练权重加载警告: {e}")
            pretrained_dict = torch.load(PRETRAINED_PATH)
            model_dict = net.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            logger.success("部分权重加载成功!")
    else:
        logger.warning(f"未找到预训练权重: {PRETRAINED_PATH}")

    # 5. 定义优化器
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 6. 定义 Loss
    criterion = FaintDefectLoss(alpha=0.3, beta=0.7, gamma=2.0)

    # 7. 初始化 TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        run_name = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 日志目录: {log_dir}")
        logger.info(f"启动 TensorBoard 命令: tensorboard --logdir={TENSORBOARD_LOG_DIR}")

    # 8. 训练循环
    ite_num = 0
    running_loss = 0.0
    best_loss = float('inf')

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"开始训练 - 项目: {PROJECT_NAME}")
    logger.info(f"模型: {MODEL_NAME}, Epochs: {EPOCH_NUM}, Batch: {BATCH_SIZE}")
    logger.info(f"输入尺寸: {INPUT_SIZE}")
    logger.info(f"模型保存目录: {MODEL_SAVE_DIR}")
    logger.info("=" * 60)
    
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(EPOCH_NUM):
        net.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_target_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            inputs, labels = data["image"], data["label"]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            # Loss 计算
            loss2, loss = muti_loss_fusion(
                criterion, d0, d1, d2, d3, d4, d5, d6, labels
            )

            # Backward
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            
            optimizer.step()

            current_loss = loss.item()
            current_target_loss = loss2.item()
            running_loss += current_loss
            epoch_loss += current_loss
            epoch_target_loss += current_target_loss
            epoch_batches += 1

            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/train_iter', current_loss, ite_num)
                writer.add_scalar('Loss/target_iter', current_target_loss, ite_num)

            if ite_num % 50 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{EPOCH_NUM}, Ite {ite_num}] Loss: {running_loss/50:.4f}, LR: {LEARNING_RATE:.2e}"
                )
                running_loss = 0.0

        # Epoch 结束处理
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        avg_epoch_target_loss = epoch_target_loss / epoch_batches if epoch_batches > 0 else 0
        
        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            writer.add_scalar('Loss/target_epoch', avg_epoch_target_loss, epoch + 1)
            writer.add_scalar('Time/epoch_seconds', epoch_duration, epoch + 1)
            writer.add_scalar('Learning_Rate', LEARNING_RATE, epoch + 1)
        
        # 打印 Epoch 信息
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = EPOCH_NUM - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        logger.info(f"")
        logger.info(f"=== Epoch {epoch + 1}/{EPOCH_NUM} 完成 ===")
        logger.info(f"    平均Loss: {avg_epoch_loss:.6f} (Target: {avg_epoch_target_loss:.6f})")
        logger.info(f"    当前LR: {LEARNING_RATE:.2e}")
        logger.info(f"    本Epoch耗时: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)")
        logger.info(f"    预计剩余时间: {estimated_remaining/60:.1f}min ({estimated_remaining/3600:.2f}h)")

        # 定期保存模型
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_epoch_{epoch+1}.pth")
            torch.save(net.state_dict(), save_path)
            logger.info(f"模型已保存: {save_path}")

    # 训练结束
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.success("训练完成!")
    logger.info("=" * 60)
    logger.info(f"项目名称: {PROJECT_NAME}")
    logger.info(f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}min, {total_duration/3600:.2f}h)")
    logger.info(f"模型保存目录: {MODEL_SAVE_DIR}")
    
    if writer is not None:
        writer.close()
        logger.info(f"TensorBoard 日志: {TENSORBOARD_LOG_DIR}")


# =====================================================================
#                           主入口
# =====================================================================

def main():
    """主入口: 自动转换数据 + 训练"""
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"U2Net 一体化训练 - 项目: {PROJECT_NAME}")
    logger.info("=" * 60)
    logger.info(f"源数据目录: {SOURCE_DATA_DIR}")
    logger.info(f"项目目录: {PROJECT_DIR}")
    logger.info("")
    
    # Step 1: 转换数据
    if not convert_dataset():
        logger.error("数据转换失败，训练中止")
        return
    
    # Step 2: 开始训练
    train()


if __name__ == "__main__":
    main()
