import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
from datetime import datetime

# ======= 可视化工具配置 =======
# 选择使用的可视化工具: 'tensorboard', 'wandb', 'both', 或 'none'
VISUALIZATION_TOOL = 'tensorboard'  # TensorBoard 不需要 Docker，更简单

# wandb 配置 (如果使用 wandb)
WANDB_PROJECT = 'u2net-training'  # wandb 项目名称
WANDB_RUN_NAME = None  # 运行名称，None 则自动生成
WANDB_OFFLINE = True   # 离线模式，日志保存到本地 wandb/ 目录

# TensorBoard 日志目录
TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), 'runs')
# =============================

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# Weights & Biases (更强大的可视化工具，可选)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    if VISUALIZATION_TOOL in ['wandb', 'both']:
        print("Warning: wandb not available. Install with: pip install wandb")

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------
# 使用针对淡缺陷的 Focal Loss + IoU Loss
from losses import FaintDefectLoss, muti_loss_fusion

# 初始化Loss
# positive_weight: 缺陷像素的权重，缺陷越少越大(建议5-20)
# gamma: Focal Loss参数，越大越关注难分类样本
criterion = FaintDefectLoss(focal_weight=1.0, iou_weight=0.5, positive_weight=15.0, gamma=2.0)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """兼容原有代码的包装函数"""
    loss0, total_loss = muti_loss_fusion(criterion, d0, d1, d2, d3, d4, d5, d6, labels_v)
    print(f"loss0: {loss0.data.item():.6f}, total: {total_loss.data.item():.6f}")
    return loss0, total_loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data', 'daowenb402' + os.sep)
tra_image_dir = 'images' + os.sep
tra_label_dir = 'masks' + os.sep

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

# ======= 用户可调参数 =======
epoch_num =100 # finetune建议12-24个epoch，可根据loss变化调整
save_every_n_epochs = 10 # 每隔几个epoch保存一次模型
batch_size_train = 8  # 根据显存调整，如果OOM可以改小
# ============================

batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

# 自动计算每个epoch的iteration数和保存频率
iterations_per_epoch = (train_num + batch_size_train - 1) // batch_size_train  # 向上取整
save_frq = iterations_per_epoch * save_every_n_epochs  # 每N个epoch保存一次
print(f"Iterations per epoch: {iterations_per_epoch}")
print(f"Will save model every {save_every_n_epochs} epochs ({save_frq} iterations)")
print(f"Total epochs: {epoch_num}, Total iterations: {epoch_num * iterations_per_epoch}")
print("---")

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)  # Windows上用0避免多进程问题

if __name__ == '__main__':
    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 3.5 load pretrained weights (finetune) --------
    pretrained_model_path = os.path.join(os.getcwd(), 'saved_models', 'u2net', 'u2net.pth')
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from: {pretrained_model_path}")
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(pretrained_model_path))
        else:
            net.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        print("Pretrained model loaded successfully!")
    else:
        print(f"No pretrained model found at {pretrained_model_path}, training from scratch...")

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # finetune用更小的学习率

    # ------- 5. training process --------
    print("---start training...")
    
    # ======= 初始化可视化工具 =======
    writer = None
    use_tensorboard = VISUALIZATION_TOOL in ['tensorboard', 'both'] and TENSORBOARD_AVAILABLE
    use_wandb = VISUALIZATION_TOOL in ['wandb', 'both'] and WANDB_AVAILABLE
    
    if use_tensorboard:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard initialized! Log dir: {log_dir}")
        print(f"启动 TensorBoard 命令: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    
    if use_wandb:
        # 设置离线模式
        if WANDB_OFFLINE:
            os.environ['WANDB_MODE'] = 'offline'
        
        wandb_run_name = WANDB_RUN_NAME or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=WANDB_PROJECT,
            name=wandb_run_name,
            config={
                "model": model_name,
                "epochs": epoch_num,
                "batch_size": batch_size_train,
                "learning_rate": 0.001,
                "image_size": 320,
                "crop_size": 288,
                "pretrained": os.path.exists(pretrained_model_path),
            }
        )
        print(f"wandb initialized! Project: {WANDB_PROJECT}, Run: {wandb_run_name}")
        if WANDB_OFFLINE:
            print(f"离线模式: 日志保存在 wandb/ 目录")
            print(f"查看曲线: 运行 wandb sync wandb/offline-run-xxx 上传后在线查看")
            print(f"         或者运行 wandb local 启动本地服务器查看")
    # ================================
    
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    # save_frq 已在前面根据 save_every_n_epochs 自动计算

    for epoch in range(0, epoch_num):
        net.train()
        epoch_loss = 0.0
        epoch_tar_loss = 0.0
        epoch_batches = 0

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            current_loss = loss.data.item()
            current_tar_loss = loss2.data.item()
            running_loss += current_loss
            running_tar_loss += current_tar_loss
            epoch_loss += current_loss
            epoch_tar_loss += current_tar_loss
            epoch_batches += 1

            # ======= 记录每次迭代的loss到可视化工具 =======
            if use_tensorboard and writer is not None:
                writer.add_scalar('Loss/train_iter', current_loss, ite_num)
                writer.add_scalar('Loss/target_iter', current_tar_loss, ite_num)
            
            if use_wandb:
                wandb.log({
                    'iteration': ite_num,
                    'train_loss': current_loss,
                    'target_loss': current_tar_loss,
                    'epoch': epoch + 1,
                })
            # ============================================

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        # ======= 记录每个epoch的平均loss =======
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        avg_epoch_tar_loss = epoch_tar_loss / epoch_batches if epoch_batches > 0 else 0
        
        if use_tensorboard and writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            writer.add_scalar('Loss/target_epoch', avg_epoch_tar_loss, epoch + 1)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_train_loss': avg_epoch_loss,
                'epoch_target_loss': avg_epoch_tar_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })
        
        print(f"\n=== Epoch {epoch + 1}/{epoch_num} completed. Avg loss: {avg_epoch_loss:.6f}, Avg tar: {avg_epoch_tar_loss:.6f} ===\n")
        # ========================================

    # ======= 训练结束，关闭日志记录器 =======
    print("Training finished!")
    if use_tensorboard and writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
        print(f"查看训练曲线: tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    
    if use_wandb:
        wandb.finish()
        print("wandb run finished!")
    # ======================================
