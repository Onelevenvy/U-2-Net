"""
针对微小淡缺陷的组合Loss函数
- Focal Loss: 解决样本不平衡，关注淡缺陷
- IoU Loss: 解决小目标漏检
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaintDefectLoss(nn.Module):
    """
    针对淡缺陷检测的组合Loss
    
    Args:
        focal_weight: Focal Loss的权重
        iou_weight: IoU Loss的权重
        positive_weight: 正样本(缺陷)的权重，缺陷越少设置越大(5-20)
        gamma: Focal Loss的gamma参数，越大越关注难分类样本
    """
    def __init__(self, focal_weight=1.0, iou_weight=0.5, positive_weight=10.0, gamma=2.0):
        super(FaintDefectLoss, self).__init__()
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.positive_weight = positive_weight
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 模型输出 (已经过sigmoid)
            target: [B, 1, H, W] 标签
        """
        # 确保数值稳定
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # 1. Focal Loss (针对淡缺陷的核心)
        # BCE部分
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # p_t: 正确分类的置信度
        p_t = pred * target + (1 - pred) * (1 - target)
        
        # Focal weight: (1 - p_t)^gamma，难分类样本权重更大
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        # 正负样本权重: 缺陷像素(正样本)权重更大
        sample_weights = target * self.positive_weight + (1 - target)
        focal_loss = (focal_loss * sample_weights).mean()

        # 2. IoU Loss (针对小目标不消失)
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_loss = 1 - iou.mean()

        total_loss = self.focal_weight * focal_loss + self.iou_weight * iou_loss
        
        return total_loss


def muti_loss_fusion(criterion, d0, d1, d2, d3, d4, d5, d6, labels):
    """
    U2Net深监督Loss融合
    每一个尺度都要算Loss，d0是最终结果权重最大
    
    Args:
        criterion: Loss函数实例
        d0-d6: 网络7个输出
        labels: 标签
    
    Returns:
        loss0: 主输出的loss (用于打印)
        total_loss: 总loss (用于反向传播)
    """
    loss0 = criterion(d0, labels)
    loss1 = criterion(d1, labels)
    loss2 = criterion(d2, labels)
    loss3 = criterion(d3, labels)
    loss4 = criterion(d4, labels)
    loss5 = criterion(d5, labels)
    loss6 = criterion(d6, labels)

    # d0 是最终融合结果，权重最大
    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, total_loss
