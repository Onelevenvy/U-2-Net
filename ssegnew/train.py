
import os
import json
import time
import glob
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
from torchvision import transforms
from loguru import logger

# 导入拆分后的模块
import config as cfg
import data_prep
from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import muti_loss_fusion, get_loss_function

cudnn.benchmark = True

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
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # 获取图片和标签文件列表
    image_ext = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        if len(glob.glob(os.path.join(cfg.IMAGES_DIR, f"*{ext}"))) > 0:
            image_ext = ext
            break

    if image_ext is None:
        logger.error("未找到训练图片！")
        return

    tra_img_name_list = glob.glob(os.path.join(cfg.IMAGES_DIR, f"*{image_ext}"))
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(cfg.MASKS_DIR, f"{base_name}.png")
        tra_lbl_name_list.append(lbl_path)

    logger.info(f"训练图片数量: {len(tra_img_name_list)}")

    # 2. 定义 DataLoader
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [
                RescaleT(cfg.INPUT_SIZE),
                CLAHE_Transform(),
                ToTensorLab(flag=0, num_classes=cfg.NUM_CLASSES),
            ]
        ),
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        drop_last=True,
    )
    logger.info(f"DataLoader: batch_size={cfg.BATCH_SIZE}, num_workers={cfg.NUM_WORKERS}")

    # 3. 定义模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.MODEL_NAME == "u2net":
        net = U2NET(3, cfg.NUM_CLASSES)
    elif cfg.MODEL_NAME == "u2netp":
        net = U2NETP(3, cfg.NUM_CLASSES)
    else:
        logger.error(f"未知模型: {cfg.MODEL_NAME}")
        return

    net = net.to(device)
    logger.info(f"模型 {cfg.MODEL_NAME} 加载到设备: {device}")
    logger.info(
        f"输出类别数: {cfg.NUM_CLASSES} ({'二值分割' if cfg.NUM_CLASSES == 1 else '多类别分割'})"
    )

    # 4. 加载预训练权重 (必须在 DataParallel 包装之前)
    if os.path.exists(cfg.PRETRAINED_PATH):
        logger.info(f"加载预训练权重: {cfg.PRETRAINED_PATH}")
        try:
            pretrained_dict = torch.load(cfg.PRETRAINED_PATH)
            model_dict = net.state_dict()

            # 过滤掉不匹配的层 (主要是输出层如果类别数不同)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            loaded_keys = len(pretrained_dict)
            total_keys = len(model_dict)

            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            logger.success(f"预训练权重加载成功! ({loaded_keys}/{total_keys} 层)")

            if cfg.NUM_CLASSES > 1 and loaded_keys < total_keys:
                logger.warning("由于类别数改变，side 层和 outconv 层需要重新训练")

        except Exception as e:
            logger.warning(f"预训练权重加载失败: {e}")
    else:
        logger.warning(f"未找到预训练权重: {cfg.PRETRAINED_PATH}")

    # 5. 多卡并行训练 (必须在加载预训练权重之后)
    if cfg.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        logger.info(f"启用多卡并行训练: 使用 {torch.cuda.device_count()} 张 GPU")
        net = nn.DataParallel(net)
        # 多卡时建议按GPU数量调整batch_size
        effective_batch_size = cfg.BATCH_SIZE * torch.cuda.device_count()
        logger.info(
            f"有效 Batch Size: {effective_batch_size} (单卡 {cfg.BATCH_SIZE} x {torch.cuda.device_count()} GPUs)"
        )
    else:
        logger.info(f"使用单卡训练: {device}")

    # 5. 定义优化器
    optimizer = optim.AdamW(net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # 定义 LR 调度器 (Cosine Annealing) - 依据配置决定是否启动
    scheduler = None
    if cfg.USE_COSINE_ANNEALING:
        logger.info("启用 CosineAnnealingWarmRestarts 调度器")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.COSINE_T0, T_mult=cfg.COSINE_TMULT
        )
    else:
        logger.info("禁用 CosineAnnealingWarmRestarts 调度器 (使用恒定 LR)")

    # 6. 定义 Loss (自动根据类别数选择)
    criterion = get_loss_function(cfg.NUM_CLASSES, cfg.CLASS_WEIGHTS)
    logger.info(f"损失函数: {criterion.__class__.__name__}")

    # 7. 初始化 TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        run_name = f"{cfg.MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(cfg.TENSORBOARD_LOG_DIR, run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 日志目录: {log_dir}")
        logger.info(
            f"启动 TensorBoard 命令: tensorboard --logdir={cfg.TENSORBOARD_LOG_DIR}"
        )

    # 8. 训练循环
    ite_num = 0
    running_loss = 0.0
    
    # ======= 训练开始时保存配置文件 =======
    config_path = os.path.join(cfg.MODEL_SAVE_DIR, "config.json")
    gpu_count = torch.cuda.device_count() if cfg.USE_MULTI_GPU else 1
    config_dict = {
        "project_name": cfg.PROJECT_NAME,
        "model_name": cfg.MODEL_NAME,
        "input_size": list(cfg.INPUT_SIZE),  # (H, W)
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES if cfg.NUM_CLASSES > 1 else {},
        "use_clahe": True,
        "batch_size": cfg.BATCH_SIZE,
        "use_multi_gpu": cfg.USE_MULTI_GPU,
        "gpu_count": gpu_count,
        "effective_batch_size": cfg.BATCH_SIZE * gpu_count,
        "learning_rate": cfg.LEARNING_RATE,
        "total_epochs": cfg.EPOCH_NUM,
        "use_cosine_annealing": cfg.USE_COSINE_ANNEALING,
        "best_model": None,  # 还没有保存的模型
        "last_saved_epoch": 0,
        "train_time_seconds": 0,
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"配置文件已创建: {config_path}")

    total_start_time = time.time()
    epoch_times = []

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"开始训练 - 项目: {cfg.PROJECT_NAME}")
    logger.info(f"模型: {cfg.MODEL_NAME}, Epochs: {cfg.EPOCH_NUM}, Batch: {cfg.BATCH_SIZE}")
    logger.info(f"输入尺寸: {cfg.INPUT_SIZE}")
    logger.info(f"模型保存目录: {cfg.MODEL_SAVE_DIR}")
    logger.info("=" * 60)

    for epoch in range(cfg.EPOCH_NUM):
        net.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_target_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(tqdm(salobj_dataloader)):
            ite_num += 1
            inputs, labels = data["image"], data["label"]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward (Normal FP32)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            # Loss 计算
            loss2, loss = muti_loss_fusion(
                criterion, d0, d1, d2, d3, d4, d5, d6, labels
            )

            # Backward
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

            # Optimizer Step
            optimizer.step()
            

            current_loss = loss.item()
            current_target_loss = loss2.item()
            running_loss += current_loss
            epoch_loss += current_loss
            epoch_target_loss += current_target_loss
            epoch_batches += 1

            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/train_iter", current_loss, ite_num)
                writer.add_scalar("Loss/target_iter", current_target_loss, ite_num)

            if ite_num % 50 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{cfg.EPOCH_NUM}, Ite {ite_num}] Loss: {running_loss/50:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                running_loss = 0.0

        # Epoch 结束处理
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # [Scheduler] 更新学习率
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]

        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        avg_epoch_target_loss = (
            epoch_target_loss / epoch_batches if epoch_batches > 0 else 0
        )

        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch + 1)
            writer.add_scalar("Loss/target_epoch", avg_epoch_target_loss, epoch + 1)
            writer.add_scalar("Time/epoch_seconds", epoch_duration, epoch + 1)
            writer.add_scalar("Learning_Rate", current_lr, epoch + 1)

        # 打印 Epoch 信息
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = cfg.EPOCH_NUM - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs

        logger.info(f"")
        logger.info(f"=== Epoch {epoch + 1}/{cfg.EPOCH_NUM} 完成 ===")
        logger.info(
            f"    平均Loss: {avg_epoch_loss:.6f} (Target: {avg_epoch_target_loss:.6f})"
        )
        logger.info(f"    当前LR: {current_lr:.2e}")
        logger.info(
            f"    本Epoch耗时: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)"
        )
        logger.info(
            f"    预计剩余时间: {estimated_remaining/60:.1f}min ({estimated_remaining/3600:.2f}h)"
        )

        # 定期保存模型
        if (epoch + 1) % cfg.SAVE_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(
                cfg.MODEL_SAVE_DIR, f"{cfg.MODEL_NAME}_epoch_{epoch+1}.pth"
            )
            # 处理 DataParallel 包装: 保存原始模型参数
            model_to_save = net.module if hasattr(net, "module") else net
            torch.save(model_to_save.state_dict(), save_path)
            logger.info(f"模型已保存: {save_path}")

            # 更新配置文件
            config_dict["best_model"] = f"{cfg.MODEL_NAME}_epoch_{epoch+1}.pth"
            config_dict["last_saved_epoch"] = epoch + 1
            config_dict["train_time_seconds"] = time.time() - total_start_time
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)

    # 训练结束
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # ======= 更新最终配置 =======
    config_dict["train_time_seconds"] = total_duration
    config_dict["train_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"配置文件已更新: {config_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.success("训练完成!")
    logger.info("=" * 60)
    logger.info(f"项目名称: {cfg.PROJECT_NAME}")
    logger.info(
        f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}min, {total_duration/3600:.2f}h)"
    )
    logger.info(f"模型保存目录: {cfg.MODEL_SAVE_DIR}")

    if writer is not None:
        writer.close()
        logger.info(f"TensorBoard 日志: {cfg.TENSORBOARD_LOG_DIR}")


# =====================================================================
#                           主入口
# =====================================================================


def main():
    """主入口: 自动转换数据 + 训练"""

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"U2Net 一体化训练 - 项目: {cfg.PROJECT_NAME}")
    logger.info("=" * 60)
    logger.info(f"源数据目录: {cfg.SOURCE_DATA_DIR}")
    logger.info(f"项目目录: {cfg.PROJECT_DIR}")
    logger.info("")

    # Step 1: 转换数据
    if not data_prep.convert_dataset():
        logger.error("数据转换失败，训练中止")
        return

    # Step 2: 开始训练
    train()


if __name__ == "__main__":
    main()
