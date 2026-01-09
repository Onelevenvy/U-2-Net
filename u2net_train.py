import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
from torchvision import transforms
import glob
from datetime import datetime
import time
from loguru import logger

from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import FaintDefectLoss, muti_loss_fusion


cudnn.benchmark = True  # 固定尺寸输入时，cudnn会自动选择最优算法

TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), 'runs')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

# ======= 核心参数配置 =======
model_name = "u2netp"  #  lite 版 (u2netp)
# model_name = "u2net"  #  
batch_size_train = 8
epoch_num = 300
learning_rate = 1e-3  # 初始学习率

# 输入尺寸设置：(Height, Width)
# 原图是 2000x480，训练时建议等比例缩小到640以内

input_size = (224, 512)

data_dir = os.path.join(os.getcwd(), "train_data", "daowenb402" + os.sep)
tra_image_dir = "images" + os.sep
tra_label_dir = "masks" + os.sep
image_ext = ".jpg"
label_ext = ".png"
model_dir = os.path.join(os.getcwd(), "saved_models", model_name + os.sep)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def main():
    # 1. 数据集准备
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + "*" + image_ext)
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        lbl_name = img_name.replace(image_ext, label_ext)
        tra_lbl_name_list.append(os.path.join(data_dir, tra_label_dir, lbl_name))

    logger.info(f"Train images: {len(tra_img_name_list)}")

    # 2. 定义 DataLoader
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [
                RescaleT(input_size),
                CLAHE_Transform(), 
                ToTensorLab(flag=0),
            ]
        ),
    )

 
    # Windows 上 num_workers > 0 可能有问题，如果报错就改回 0
    num_workers = 4
    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # 加速 CPU->GPU 传输
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    logger.info(f"DataLoader: batch_size={batch_size_train}, num_workers={num_workers}")

    # 3. 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "u2net":
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        net = U2NETP(3, 1)

    net = net.to(device)
    logger.info(f"Model loaded on device: {device}")

    # 4. 加载预训练权重
    pretrained_path = os.path.join(
        os.getcwd(), "saved_models", "pretrain", f"{model_name}.pth"
    )
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained: {pretrained_path}")
        try:
            net.load_state_dict(torch.load(pretrained_path), strict=False)
        except Exception as e:
            logger.warning(f"Pretrained load warning: {e}")
            pretrained_dict = torch.load(pretrained_path)
            model_dict = net.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            logger.success("Partial weights loaded!")
    else:
        logger.warning("No pretrained weights found! Training will be slow.")

    # 5. 定义优化器 (AdamW 带权重衰减)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # ======= 学习率调度器  =======
    # CosineAnnealingWarmRestarts: 周期性重启，避免陷入局部最优
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    # logger.info("Scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)")

    # 6. 定义 Loss
    criterion = FaintDefectLoss(alpha=0.3, beta=0.7, gamma=2.0)

    # ======= 初始化 TensorBoard =======
    writer = None
    if TENSORBOARD_AVAILABLE:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_name)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard initialized! Log dir: {log_dir}")
        logger.info(f"启动 TensorBoard 命令: tensorboard --logdir={TENSORBOARD_LOG_DIR}")

    # 7. 训练循环 (保持原始逻辑，不用 AMP)
    ite_num = 0
    running_loss = 0.0
    best_loss = float('inf')

    logger.info("Start Training (FP32 模式，确保收敛)")
    
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(epoch_num):
        net.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_target_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            inputs, labels = data["image"], data["label"]

            # 使用 non_blocking 加速传输
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 标准训练流程 (不使用 AMP，确保收敛)
            optimizer.zero_grad()

            # Forward
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            # Loss 计算
            loss2, loss = muti_loss_fusion(
                criterion, d0, d1, d2, d3, d4, d5, d6, labels
            )

            # Backward (标准方式)
            loss.backward()
            
            # 梯度裁剪 (宽松阈值，只防止极端情况)
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
                    f"[Epoch {epoch+1}/{epoch_num}, Ite {ite_num}] Loss: {running_loss/50:.4f}, LR: {learning_rate:.2e}"
                )
                running_loss = 0.0

        # ======= Epoch 结束处理 =======
        # 更新学习率
        # scheduler.step()
        
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
            writer.add_scalar('Learning_Rate', learning_rate, epoch + 1)
        
        # 打印 Epoch 信息
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epoch_num - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        logger.info(f"")
        logger.info(f"=== Epoch {epoch + 1}/{epoch_num} 完成 ===")
        logger.info(f"    平均Loss: {avg_epoch_loss:.6f} (Target: {avg_epoch_target_loss:.6f})")
        logger.info(f"    当前LR: {learning_rate:.2e}")
        logger.info(f"    本Epoch耗时: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)")
        logger.info(f"    预计剩余时间: {estimated_remaining/60:.1f}min ({estimated_remaining/3600:.2f}h)")

        # # 保存最佳模型
        # if avg_epoch_loss < best_loss:
        #     best_loss = avg_epoch_loss
        #     best_path = f"{model_dir}{model_name}_best.pth"
        #     torch.save(net.state_dict(), best_path)
        #     logger.success(f"🏆 New best model saved: {best_path} (loss: {best_loss:.6f})")

        # 每 10 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = f"{model_dir}{model_name}_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), save_path)
            logger.info(f"Checkpoint saved: {save_path}")

    # ======= 训练结束 =======
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    logger.info("")
    logger.info("=" * 50)
    logger.success("训练完成!")
    logger.info("=" * 50)
    logger.info(f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}min, {total_duration/3600:.2f}h)")
    logger.info(f"最佳Loss: {best_loss:.6f}")
    
    if writer is not None:
        writer.close()
        logger.info(f"TensorBoard logs: {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    main()
