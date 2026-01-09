import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#                    二值分割损失函数 (num_classes=1)
# =====================================================================

class FaintDefectLoss(nn.Module):
    """
    针对淡缺陷优化的二值分割损失函数
    使用 Tversky + Focal Loss 组合
    """
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


# =====================================================================
#                    多类别分割损失函数 (num_classes>1)
# =====================================================================

class MultiClassLoss(nn.Module):
    """
    多类别分割损失函数
    使用 CrossEntropy + Dice Loss 组合
    
    Args:
        num_classes: 类别数 (包含背景)
        class_weights: 各类别权重 (可选)，用于处理类别不平衡
        dice_weight: Dice Loss 权重 (默认0.5)
        ce_weight: CrossEntropy 权重 (默认0.5)
        ignore_index: 忽略的标签索引 (默认-100)
    """
    def __init__(self, num_classes, class_weights=None, dice_weight=0.5, ce_weight=0.5, ignore_index=-100):
        super(MultiClassLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        
        # CrossEntropy Loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def dice_loss(self, pred, target):
        """
        计算多类别 Dice Loss
        pred: [B, C, H, W] softmax 后的概率图
        target: [B, H, W] 类别索引图
        """
        smooth = 1.0
        
        # one-hot 编码 target: [B, H, W] -> [B, C, H, W]
        target_one_hot = F.one_hot(target.long(), self.num_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # 计算每个类别的 Dice
        dice_per_class = []
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :]
            target_c = target_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_per_class.append(dice)
        
        # 平均 Dice Loss
        mean_dice = sum(dice_per_class) / len(dice_per_class)
        return 1.0 - mean_dice
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] - softmax 后的概率图
        target: [B, 1, H, W] 或 [B, H, W] - 类别索引图 (0, 1, 2, ...)
        """
        # 处理 target 维度
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        target = target.long()
        
        # 1. CrossEntropy Loss
        # CrossEntropyLoss 需要 pred 是 logits，但我们的输入已经是 softmax 后的
        # 所以需要转回 logits (取 log)
        eps = 1e-6
        pred_clamped = torch.clamp(pred, min=eps, max=1.0 - eps)
        logits = torch.log(pred_clamped)
        ce = self.ce_loss(logits, target)
        
        # 2. Dice Loss
        dice = self.dice_loss(pred, target)
        
        # 3. 组合
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss


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
    根据类别数自动选择损失函数
    
    Args:
        num_classes: 类别数
        class_weights: 类别权重 (仅多类别时有效)
    
    Returns:
        criterion: 损失函数实例
    """
    if num_classes == 1:
        return FaintDefectLoss(alpha=0.3, beta=0.7, gamma=2.0)
    else:
        return MultiClassLoss(num_classes=num_classes, class_weights=class_weights)