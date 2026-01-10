"""
U2Net 项目化推理脚本
- 只需指定项目名称，自动读取训练配置
- 结果保存在项目文件夹的 predictions/ 目录下

用法:
    python predict.py
"""
import os

# 设置可用的 GPU 卡 (必须在 import torch 之前设置)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import os
import json
import glob
import time
import cv2
import torch
import numpy as np
from loguru import logger

from model import U2NET, U2NETP


# =====================================================================
#                           用户配置区域
# =====================================================================

# 1. 项目名称 (与训练时的 PROJECT_NAME 保持一致)
PROJECT_NAME = "xmy2"

# 2. 使用哪个模型 (留空则自动使用 config.json 中的 best_model)
MODEL_FILE = ""  # 例如 "u2netp_epoch_200.pth"，留空自动选择

# 3. 测试图片目录
TEST_IMAGE_DIR = (
   r"\\192.168.1.55\ai研究院\5_临时文件夹\czj\1.datatest\2_新美洋\2_Skolpha\1_train\100pcs"
)
# TEST_IMAGE_DIR = r"\\192.168.1.55\ai研究院\5_临时文件夹\czj\1.datatest\2_新美洋\2_Skolpha\2_test\1_画线+膜破"


# 4. 可视化配置
OVERLAY_COLOR = (0, 0, 255)  # 红色 (BGR格式) - 仅二值分割使用
MAX_ALPHA = 0.7  # 最大透明度

# 多类别可视化颜色表 (BGR格式)
# 索引 0 = 背景 (不显示), 1/2/3... = 不同类别
CLASS_COLORS = [
    (0, 0, 0),  # 0: 背景 - 不显示
    (0, 0, 255),  # 1: 红色
    (0, 255, 0),  # 2: 绿色
    (255, 0, 0),  # 3: 蓝色
    (0, 255, 255),  # 4: 黄色
    (255, 0, 255),  # 5: 紫色
    (255, 255, 0),  # 6: 青色
    (128, 128, 255),  # 7: 淡红色
    (128, 255, 128),  # 8: 淡绿色
]

# =====================================================================
#                           自动计算路径
# =====================================================================

PROJECT_DIR = os.path.join(os.getcwd(), "projects", PROJECT_NAME)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "predictions")


# =====================================================================
#                           工具函数
# =====================================================================


def load_config():
    """加载项目配置"""
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"配置文件不存在: {CONFIG_PATH}")
        logger.error("请先运行 train.py 训练模型！")
        return None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"已加载配置: {CONFIG_PATH}")
    logger.info(f"  模型: {config['model_name']}")
    logger.info(f"  输入尺寸: {config['input_size']}")
    logger.info(f"  训练日期: {config.get('train_date', 'N/A')}")

    return config


def cv2_read_img(file_path):
    """解决 Windows 下中文路径读取问题"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def preprocess_image(image_path, scale, use_clahe=True):
    """
    预处理流水线 - 复用 data_loader 的 transforms
    
    Args:
        image_path: 图片路径
        scale: tuple (max_long, min_short) 或 int
        use_clahe: 是否使用 CLAHE 增强
    
    Returns:
        img_tensor: [1, 3, H, W] 预处理后的张量
        original_shape: (h, w) 原始图像尺寸
        img_bgr: 原始 BGR 图像
        resized_shape: (h, w) resize 后的尺寸 (padding 前)
        padded_shape: (h, w) padding 后的尺寸
    """
    from data_loader import RescaleT, PadToMultiple, CLAHE_Transform, ToTensorLab
    from torchvision import transforms
    
    img_bgr = cv2_read_img(image_path)
    if img_bgr is None:
        return None, None, None, None, None

    # 灰度图转3通道
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    original_shape = img_bgr.shape[:2]  # (h, w)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 构建 transform pipeline (复用 data_loader 的类)
    transform_list = [
        RescaleT(scale),
        PadToMultiple(divisor=32),
    ]
    if use_clahe:
        transform_list.append(CLAHE_Transform())
    transform_list.append(ToTensorLab(flag=0, num_classes=1))  # num_classes 不影响图像处理
    
    transform = transforms.Compose(transform_list)
    
    # 构造 sample (模拟 Dataset 的格式)
    # 推理时 label 可以用空的
    sample = {
        "imidx": np.array([0]),
        "image": img_rgb,
        "label": np.zeros(original_shape, dtype=np.uint8)
    }
    
    # 应用 transform
    sample = transform(sample)
    
    # 获取 tensor 和相关信息
    img_tensor = sample["image"].unsqueeze(0)  # [1, C, H, W]
    
    # 计算 resized_shape 和 padded_shape (使用 MMDet 风格逻辑)
    h, w = original_shape
    long_side = max(h, w)
    short_side = min(h, w)
    
    # 解析 scale 参数
    if isinstance(scale, int):
        max_long, min_short = scale, scale
    else:
        max_long, min_short = scale
    
    # 按短边缩放
    ratio = min_short / short_side
    # 检查长边是否超限
    if long_side * ratio > max_long:
        ratio = max_long / long_side
    
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized_shape = (new_h, new_w)
    
    # padding 后的尺寸
    divisor = 32
    padded_h = int(np.ceil(new_h / divisor)) * divisor
    padded_w = int(np.ceil(new_w / divisor)) * divisor
    padded_shape = (padded_h, padded_w)

    return img_tensor, original_shape, img_bgr, resized_shape, padded_shape


def load_model(config):
    """根据配置加载模型"""
    model_name = config["model_name"]
    num_classes = config.get("num_classes", 1)

    # 确定模型文件
    if MODEL_FILE:
        model_file = MODEL_FILE
    else:
        model_file = config.get(
            "best_model", f"{model_name}_epoch_{config['total_epochs']}.pth"
        )

    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None

    logger.info(f"加载模型: {model_path}")
    logger.info(
        f"类别数: {num_classes} ({'二值分割' if num_classes == 1 else '多类别分割'})"
    )

    # 实例化网络
    if model_name == "u2net":
        net = U2NET(3, num_classes)
    elif model_name == "u2netp":
        net = U2NETP(3, num_classes)
    else:
        logger.error(f"未知模型类型: {model_name}")
        return None

    # 加载权重
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))

    net.eval()
    return net


def predict(net, img_tensor, num_classes=1):
    """
    模型推理

    Returns:
        num_classes=1: 返回概率图 [H, W], 值在 0-1 之间
        num_classes>1: 返回类别索引图 [H, W], 值为 0,1,2,...
    """
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        d0, *_ = net(img_tensor)

        if num_classes == 1:
            # 二值分割: 返回概率图
            pred = d0[:, 0, :, :]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            return pred.cpu().numpy().squeeze()
        else:
            # 多类别分割: 返回类别索引
            pred = torch.argmax(d0, dim=1)  # [B, H, W]
            return pred.cpu().numpy().squeeze()


def draw_labelme_annotations(img, json_path):
    """在图像上绘制 labelme 标注 (如果存在)"""
    if not os.path.exists(json_path):
        return img

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return img

    img_draw = img.copy()
    ANNOTATION_COLOR = (0, 255, 0)  # 绿色
    LINE_THICKNESS = 2

    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if not points:
            continue

        pts = np.array(points, dtype=np.int32)

        if shape_type == "polygon":
            cv2.polylines(
                img_draw,
                [pts],
                isClosed=True,
                color=ANNOTATION_COLOR,
                thickness=LINE_THICKNESS,
            )
        elif shape_type == "rectangle" and len(pts) >= 2:
            cv2.rectangle(
                img_draw,
                tuple(pts[0]),
                tuple(pts[1]),
                color=ANNOTATION_COLOR,
                thickness=LINE_THICKNESS,
            )
        elif shape_type == "circle" and len(pts) >= 2:
            center = tuple(pts[0])
            radius = int(np.linalg.norm(pts[0] - pts[1]))
            cv2.circle(
                img_draw,
                center,
                radius,
                color=ANNOTATION_COLOR,
                thickness=LINE_THICKNESS,
            )

    return img_draw


def overlay_result(
    original_img, pred_mask, output_path, img_path=None, num_classes=1, class_names=None,
    resized_shape=None
):
    """
    生成可视化结果

    Args:
        original_img: 原图 BGR
        pred_mask: 预测结果 (可能包含 padding 区域)
            - num_classes=1: 概率图 [H, W], 0-1
            - num_classes>1: 类别索引图 [H, W], 0,1,2,...
        output_path: 输出路径
        img_path: 原图路径 (用于查找 GT 标注)
        num_classes: 类别数
        class_names: 类别名称映射 (dict)
        resized_shape: (h, w) resize 后的尺寸 (padding 前), 用于裁剪 padding
    """
    h, w = original_img.shape[:2]

    # 裁剪掉 padding 区域 (如果有)
    if resized_shape is not None:
        rh, rw = resized_shape
        pred_mask = pred_mask[:rh, :rw]

    # 还原 mask 到原图尺寸
    if num_classes == 1:
        # 二值分割: 用概率值作为透明度
        mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap = np.zeros_like(original_img)
        heatmap[:] = OVERLAY_COLOR

        alpha = mask_resized * MAX_ALPHA
        alpha[alpha < 0.1] = 0
        alpha = np.stack([alpha] * 3, axis=-1)

        overlay = original_img * (1 - alpha) + heatmap * alpha
        overlay = overlay.astype(np.uint8)
    else:
        # 多类别分割: 为每个类别分配不同颜色
        mask_resized = cv2.resize(
            pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

        overlay = original_img.copy()
        for class_idx in range(1, num_classes):  # 跳过背景 (0)
            if class_idx < len(CLASS_COLORS):
                color = CLASS_COLORS[class_idx]
            else:
                color = (128, 128, 128)  # 默认灰色

            mask_class = mask_resized == class_idx
            if mask_class.any():
                # 叠加颜色
                for c in range(3):
                    overlay[:, :, c][mask_class] = (
                        original_img[:, :, c][mask_class] * (1 - MAX_ALPHA)
                        + color[c] * MAX_ALPHA
                    ).astype(np.uint8)

    # 原图 (带标注)
    original_with_annotation = original_img.copy()
    if img_path:
        base_path = os.path.splitext(img_path)[0]
        json_path = base_path + ".json"
        original_with_annotation = draw_labelme_annotations(
            original_with_annotation, json_path
        )

    # 拼接: 上=预测结果, 下=原图(带GT标注)
    combined = np.vstack([overlay, original_with_annotation])
    cv2.imencode(".jpg", combined)[1].tofile(output_path)


# =====================================================================
#                           主函数
# =====================================================================


def main():
    # 1. 加载配置
    config = load_config()
    if config is None:
        return

    # 读取配置
    input_scale = config.get("input_scale", config.get("input_size", 512))  # 兼容旧配置
    use_clahe = config.get("use_clahe", True)
    num_classes = config.get("num_classes", 1)
    class_names = config.get("class_names", {})

    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 加载模型
    net = load_model(config)
    if net is None:
        return

    # 4. 获取测试图片
    exts = ["*.jpg", "*.png", "*.bmp", "*.jpeg"]
    image_list = []
    for ext in exts:
        image_list.extend(glob.glob(os.path.join(TEST_IMAGE_DIR, ext)))

    if len(image_list) == 0:
        logger.error(f"未找到测试图片: {TEST_IMAGE_DIR}")
        return

    logger.info(f"找到 {len(image_list)} 张测试图片")
    logger.info(f"结果将保存到: {OUTPUT_DIR}")
    logger.info("")

    # 5. 推理
    total_time = 0
    for i, img_path in enumerate(image_list):
        fname = os.path.basename(img_path)

        # 预处理
        img_tensor, orig_shape, orig_img_bgr, resized_shape, padded_shape = preprocess_image(
            img_path, input_scale, use_clahe
        )
        if img_tensor is None:
            logger.warning(f"无法读取: {fname}")
            continue

        # 推理
        t_start = time.perf_counter()
        pred_mask = predict(net, img_tensor, num_classes)
        t_end = time.perf_counter()

        infer_time = (t_end - t_start) * 1000
        total_time += infer_time

        # 保存结果 (传入 resized_shape 用于裁剪 padding)
        save_path = os.path.join(OUTPUT_DIR, fname)
        overlay_result(
            orig_img_bgr, pred_mask, save_path, img_path, num_classes, class_names,
            resized_shape=resized_shape
        )

        logger.info(f"[{i+1}/{len(image_list)}] {fname} - {infer_time:.1f}ms")

    logger.info("")
    logger.success(f"推理完成! 平均耗时: {total_time/len(image_list):.1f}ms")
    logger.info(f"结果保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
