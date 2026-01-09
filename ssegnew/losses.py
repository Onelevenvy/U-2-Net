import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#                    统一分割损失函数 (自动适配二值/多类别)
# =====================================================================

class OptimizedSegLoss(nn.Module):
    """
    统一的分割损失函数，适用于二值分割和多类别分割
    使用 Tversky + Focal Loss 组合，针对淡缺陷和小目标优化
    
    Args:
        num_classes: 类别数
            - 1: 二值分割，输入为 sigmoid 概率 [B, 1, H, W]
            - >1: 多类别分割，输入为 softmax 概率 [B, C, H, W]
        class_weights: 各类别权重 (可选)，用于处理类别不平衡
        alpha: Tversky 中 FP 的权重 (默认0.3)
        beta: Tversky 中 FN 的权重 (默认0.7，强调召回减少漏检)
        gamma: Focal Loss 的聚焦因子 (默认2.0)
        tversky_weight: Tversky Loss 权重 (默认0.7)
        focal_weight: Focal Loss 权重 (默认0.3)
    """
    def __init__(self, num_classes=1, class_weights=None, alpha=0.3, beta=0.7, gamma=2.0,
                 tversky_weight=0.7, focal_weight=0.3):
        super(OptimizedSegLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        # 类别权重 (用于 Focal Loss)
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, pred, target):
        """
        pred: 
            - 二值: [B, 1, H, W] sigmoid 概率
            - 多类: [B, C, H, W] softmax 概率
        target:
            - 二值: [B, 1, H, W] 0/1 标签
            - 多类: [B, 1, H, W] 或 [B, H, W] 类别索引
        """
        if self.num_classes == 1:
            return self._binary_loss(pred, target)
        else:
            return self._multiclass_loss(pred, target)
    
    def _binary_loss(self, pred, target):
        """二值分割损失"""
        eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1.0 - eps)
        
        # Tversky Loss
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        tversky_loss = 1 - tversky

        # Focal Loss
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (focal_weight * bce).mean()
        
        return self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
    
    def _multiclass_loss(self, pred, target):
        """多类别分割损失"""
        # 处理 target 维度
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()
        
        eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1.0 - eps)
        
        # one-hot 编码
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Tversky Loss (只计算前景类，跳过背景 class 0)
        smooth = 1.0
        tversky_per_class = []
        for c in range(1, self.num_classes):
            pred_c = pred[:, c, :, :].contiguous().view(-1)
            target_c = target_one_hot[:, c, :, :].contiguous().view(-1)
            
            TP = (pred_c * target_c).sum()
            FP = ((1 - target_c) * pred_c).sum()
            FN = (target_c * (1 - pred_c)).sum()
            
            tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
            tversky_per_class.append(tversky)
        
        if len(tversky_per_class) > 0:
            tversky_loss = 1.0 - sum(tversky_per_class) / len(tversky_per_class)
        else:
            tversky_loss = torch.tensor(0.0, device=pred.device)
        
        # Focal Loss
        pt = pred.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        log_pt = torch.log(pt)
        
        if self.class_weights is not None:
            pixel_weights = self.class_weights[target]
            focal_loss = (-focal_weight * log_pt * pixel_weights).mean()
        else:
            focal_loss = (-focal_weight * log_pt).mean()
        
        return self.tversky_weight * tversky_loss + self.focal_weight * focal_loss


# =====================================================================
#                    保留旧类名作为别名 (向后兼容)
# =====================================================================

# 二值分割别名
FaintDefectLoss = OptimizedSegLoss

# 多类别分割别名
MultiClassLoss = OptimizedSegLoss


# =====================================================================
#                    深监督损失融合
# =====================================================================

def muti_loss_fusion(criterion, d0, d1, d2, d3, d4, d5, d6, labels_v):
    """
    深监督 Loss 计算
    同时支持二值分割和多类别分割
    """
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


def get_loss_function(num_classes, class_weights=None):
    """
    根据类别数创建统一损失函数
    
    Args:
        num_classes: 类别数 (1=二值分割, >1=多类别分割)
        class_weights: 类别权重 (可选，用于处理类别不平衡)
    
    Returns:
        criterion: OptimizedSegLoss 实例
    """
    return OptimizedSegLoss(num_classes=num_classes, class_weights=class_weights)