import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from datetime import datetime
import time

from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import FaintDefectLoss, muti_loss_fusion

# ======= TensorBoard 配置 =======
TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), 'runs')

# TensorBoard 导入
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
# ================================

# ======= 核心参数配置 =======
model_name = "u2netp"  # 强烈建议先用 lite 版 (u2netp)
batch_size_train = 8
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

    print(f"--- Train images: {len(tra_img_name_list)} ---")

    # 2. 定义 DataLoader
    # 注意 transforms 的顺序：Rescale(矩形) -> CLAHE(增强) -> ToTensor
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [
                RescaleT(input_size),  # 使用矩形尺寸 (320, 1024)
                CLAHE_Transform(),  # 物理增强淡缺陷
                ToTensorLab(flag=0),
            ]
        ),
    )

    salobj_dataloader = DataLoader(
        salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0
    )

    # 3. 定义模型
    if model_name == "u2net":
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.cuda()

    # 4. 加载预训练权重 (必须做!)
    pretrained_path = os.path.join(
        os.getcwd(), "saved_models", "u2netp.pth"
    )  # 确保你有这个文件
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained: {pretrained_path}")
        try:
            net.load_state_dict(torch.load(pretrained_path), strict=False)
        except Exception as e:
            print(f"Pretrained load warning: {e}")
            print("Try loading strictly matching keys...")
            pretrained_dict = torch.load(pretrained_path)
            model_dict = net.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            print("Partial weights loaded!")
    else:
        print("WARNING: No pretrained weights found! Training will be slow.")

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
        print(f"TensorBoard initialized! Log dir: {log_dir}")
        print(f"启动 TensorBoard 命令: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    # ===================================

    # 7. 训练循环
    ite_num = 0
    running_loss = 0.0

    print("--- Start Training ---")
    
    # 记录总训练开始时间
    total_start_time = time.time()
    epoch_times = []  # 存储每个epoch的耗时
    
    for epoch in range(epoch_num):
        net.train()
        epoch_start_time = time.time()  # 记录当前epoch开始时间
        epoch_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            inputs, labels = data["image"], data["label"]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            # Forward
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            # Loss
            loss2, loss = muti_loss_fusion(
                criterion, d0, d1, d2, d3, d4, d5, d6, labels
            )

            # Backward
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss
            epoch_loss += current_loss
            epoch_batches += 1

            # ======= 记录每次迭代的loss到TensorBoard =======
            if writer is not None:
                writer.add_scalar('Loss/train_iter', current_loss, ite_num)
                writer.add_scalar('Loss/target_iter', loss2.item(), ite_num)
            # ================================================

            if ite_num % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{epoch_num}, Ite {ite_num}] Loss: {running_loss/50:.4f}"
                )
                running_loss = 0.0

        # 计算当前epoch耗时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 计算epoch平均loss
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        
        # ======= 记录每个epoch的信息到TensorBoard =======
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            writer.add_scalar('Time/epoch_seconds', epoch_duration, epoch + 1)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        # ================================================
        
        # 打印epoch信息
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epoch_num - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print(f"\n=== Epoch {epoch + 1}/{epoch_num} 完成 ===")
        print(f"    平均Loss: {avg_epoch_loss:.6f}")
        print(f"    本Epoch耗时: {epoch_duration:.2f}s ({epoch_duration/60:.2f}min)")
        print(f"    平均Epoch耗时: {avg_epoch_time:.2f}s")
        print(f"    预计剩余时间: {estimated_remaining/60:.1f}min ({estimated_remaining/3600:.2f}h)")
        print("")

        # 每个 10 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = f"{model_dir}{model_name}_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"Model saved: {save_path}")

    # ======= 训练结束，输出统计信息 =======
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    print(f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}min, {total_duration/3600:.2f}h)")
    print(f"总Epoch数: {epoch_num}")
    print(f"平均每Epoch耗时: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"最短Epoch耗时: {min(epoch_times):.2f}s")
    print(f"最长Epoch耗时: {max(epoch_times):.2f}s")
    
    # 关闭TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\nTensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
        print(f"查看训练曲线: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    # ======================================


if __name__ == "__main__":
    main()
