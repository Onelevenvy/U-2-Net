import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

from data_loader import RescaleT, CLAHE_Transform, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP
from losses import FaintDefectLoss, muti_loss_fusion

# ======= 核心参数配置 =======
model_name = "u2netp"  # 强烈建议先用 lite 版 (u2netp)
batch_size_train = 8
epoch_num = 200  # 有预训练权重的话，200够了
learning_rate = 1e-3  # AdamW 初始学习率

# 【关键】输入尺寸设置：(Height, Width)
# 你的原图是 2000x480 (W x H)
# 为了不压扁缺陷，我们设置一个接近 1:3 或 1:4 的矩形输入
# 320 x 1024 是个不错的选择，既能被 32 整除，又保留了长宽比
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

    # 7. 训练循环
    ite_num = 0
    running_loss = 0.0

    print("--- Start Training ---")
    for epoch in range(epoch_num):
        net.train()

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

            running_loss += loss.item()

            if ite_num % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{epoch_num}, Ite {ite_num}] Loss: {running_loss/50:.4f}"
                )
                running_loss = 0.0

        # 每个 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = f"{model_dir}{model_name}_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"Model saved: {save_path}")


if __name__ == "__main__":
    main()
