import torch
import torch.nn as nn
import torch.nn.functional as F

class FaintDefectLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0):
        super(FaintDefectLoss, self).__init__()
        # beta=0.7 意味着漏检(FN)的惩罚是误检(FP)的2倍多
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: [B, 1, H, W] - 模型输出已经 sigmoid 过了
        # target: [B, 1, H, W]
        
        # 强制 clamp 到安全范围，避免 BCE 的 log(0) 问题
        eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1.0 - eps)
        
        # 1. Tversky Loss (专门解决极小目标梯度消失)
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        tversky_loss = 1 - Tversky

        # 2. Focal Loss (解决淡缺陷难分问题)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # 计算 focal 权重: (1-pt)^gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = (focal_weight * bce).mean()
        
        # 3. 组合: Tversky 负责轮廓召回，Focal 负责像素分类
        total_loss = 0.7 * tversky_loss + 0.3 * focal_loss
        
        return total_loss

def muti_loss_fusion(criterion, d0, d1, d2, d3, d4, d5, d6, labels_v):
    # 深监督 Loss 计算
    loss0 = criterion(d0, labels_v)
    loss1 = criterion(d1, labels_v)
    loss2 = criterion(d2, labels_v)
    loss3 = criterion(d3, labels_v)
    loss4 = criterion(d4, labels_v)
    loss5 = criterion(d5, labels_v)
    loss6 = criterion(d6, labels_v)

    # d0 是主输出，权重最大
    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, total_loss