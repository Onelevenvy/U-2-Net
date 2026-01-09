import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # AMP 混合精度
import torch.backends.cudnn as cudnn
from torchvision import transforms
import glob
from datetime import datetime
import time
from loguru import logger

from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import FaintDefectLoss, muti_loss_fusion

# ======= 性能优化配置 =======
cudnn.benchmark = True  # 固定尺寸输入时，cudnn会自动选择最优算法
cudnn.deterministic = False  # 允许非确定性算法以获得更好性能

# ======= TensorBoard 配置 =======
TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), 'runs')

# TensorBoard 导入
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")
# ================================

# ======= 核心参数配置 =======
model_name = "u2netp"  # 强烈建议先用 lite 版 (u2netp)
# model_name = "u2net" 
batch_size_train = 16
epoch_num = 200  # 有预训练权重的话，200够了
learning_rate = 1e-3  # AdamW 初始学习率

# 【关键】输入尺寸设置：(Height, Width)
# 你的原图是 2000x480 (W x H)
# 为了不压扁缺陷，我们设置一个接近 1:3 或 1:4 的矩形输入
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
    # 注意 transforms 的顺序：Rescale(矩形) -> CLAHE(增强) -> ToTensor
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [
                RescaleT(input_size),  # 使用矩形尺寸 (320, 1024)
                # CLAHE_Transform(),  # 物理增强淡缺陷
                ToTensorLab(flag=0),
            ]
        ),
    )

    # ======= DataLoader 性能优化 =======
    # num_workers: Windows建议4-8, Linux可以8-16
    # pin_memory: GPU训练必开，加速Host->Device传输
    # persistent_workers: 避免每个epoch重新创建worker进程
    # prefetch_factor: 每个worker预取的batch数
    num_workers = 4  # Windows推荐值
    salobj_dataloader = DataLoader(
        salobj_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # 关键！加速GPU数据传输
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # 避免最后一个小batch影响BN
    )
    logger.info(f"DataLoader: num_workers={num_workers}, pin_memory=True, prefetch_factor=2")

    # 3. 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "u2net":
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        net = U2NETP(3, 1)

    net = net.to(device)
    logger.info(f"Model loaded on device: {device}")

    # 4. 加载预训练权重 (必须做!)
    pretrained_path = os.path.join(
        os.getcwd(), "saved_models","pretrain", f"{model_name}.pth"
    )  # 确保你有这个文件
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained: {pretrained_path}")
        try:
            net.load_state_dict(torch.load(pretrained_path), strict=False)
        except Exception as e:
            logger.warning(f"Pretrained load warning: {e}")
            logger.info("Try loading strictly matching keys...")
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

    # 5. 定义优化器 (AdamW)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

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
    # ===================================

    # ======= 混合精度训练 (AMP) 配置 =======
    use_amp = torch.cuda.is_available()  # 有GPU就启用AMP
    scaler = GradScaler(enabled=use_amp)
    # device 已在模型加载时定义
    if use_amp:
        logger.info("✅ 混合精度训练 (AMP) 已启用 - 预计提速 50-100%")
    # =====================================

    # 7. 训练循环
    ite_num = 0
    running_loss = 0.0

    logger.info("Start Training")
    
    # 记录总训练开始时间
    total_start_time = time.time()
    epoch_times = []  # 存储每个epoch的耗时
    
    for epoch in range(epoch_num):
        net.train()
        epoch_start_time = time.time()  # 记录当前epoch开始时间
        epoch_loss = 0.0
        epoch_target_loss = 0.0  # 新增：累计 target loss
        epoch_batches = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            inputs, labels = data["image"], data["label"]

            # 使用 non_blocking=True 异步传输，配合 pin_memory
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # set_to_none=True 比 zero_grad() 更高效
            optimizer.zero_grad(set_to_none=True)

            # ======= AMP 混合精度前向传播 =======
            with autocast(enabled=use_amp):
                d0, d1, d2, d3, d4, d5, d6 = net(inputs)
                loss2, loss = muti_loss_fusion(
                    criterion, d0, d1, d2, d3, d4, d5, d6, labels
                )
            # ====================================

            # ======= AMP 混合精度反向传播 =======
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ====================================

            current_loss = loss.item()
            current_target_loss = loss2.item()  # 新增
            running_loss += current_loss
            epoch_loss += current_loss
            epoch_target_loss += current_target_loss  # 新增
            epoch_batches += 1

            # ======= 记录每次迭代的loss到TensorBoard =======
            if writer is not None:
                writer.add_scalar('Loss/train_iter', current_loss, ite_num)
                writer.add_scalar('Loss/target_iter', loss2.item(), ite_num)
            # ================================================

            if ite_num % 50 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{epoch_num}, Ite {ite_num}] Loss: {running_loss/50:.4f}"
                )
                running_loss = 0.0

        # 计算当前epoch耗时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 计算epoch平均loss
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        avg_epoch_target_loss = epoch_target_loss / epoch_batches if epoch_batches > 0 else 0  # 新增
        
        # ======= 记录每个epoch的信息到TensorBoard =======
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            writer.add_scalar('Loss/target_epoch', avg_epoch_target_loss, epoch + 1)  # 新增
            writer.add_scalar('Time/epoch_seconds', epoch_duration, epoch + 1)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        # ================================================
        
        # 打印epoch信息
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epoch_num - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        logger.info(f"")
        logger.info(f"=== Epoch {epoch + 1}/{epoch_num} 完成 ===")
        logger.info(f"    平均Loss: {avg_epoch_loss:.6f}")
        logger.info(f"    本Epoch耗时: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)")
        logger.info(f"    平均Epoch耗时: {avg_epoch_time:.2f}s")
        logger.info(f"    预计剩余时间: {estimated_remaining/60:.1f}min ({estimated_remaining/3600:.2f}h)")

        # 每个 10 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = f"{model_dir}{model_name}_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), save_path)
            logger.success(f"Model saved: {save_path}")

    # ======= 训练结束，输出统计信息 =======
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    logger.info("")
    logger.info("=" * 50)
    logger.success("训练完成!")
    logger.info("=" * 50)
    logger.info(f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}min, {total_duration/3600:.2f}h)")
    logger.info(f"总Epoch数: {epoch_num}")
    logger.info(f"平均每Epoch耗时: {sum(epoch_times)/len(epoch_times):.2f}s")
    logger.info(f"最短Epoch耗时: {min(epoch_times):.2f}s")
    logger.info(f"最长Epoch耗时: {max(epoch_times):.2f}s")
    
    # 关闭TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info(f"TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
        logger.info(f"查看训练曲线: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    # ======================================


if __name__ == "__main__":
    main()
