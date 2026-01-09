"""
U2Net 项目化推理脚本
- 只需指定项目名称，自动读取训练配置
- 结果保存在项目文件夹的 predictions/ 目录下

用法:
    python predict.py
"""

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
PROJECT_NAME = "daowen_b402"

# 2. 使用哪个模型 (留空则自动使用 config.json 中的 best_model)
MODEL_FILE = ""  # 例如 "u2netp_epoch_200.pth"，留空自动选择

# 3. 测试图片目录
TEST_IMAGE_DIR = r"F:\New_SourceCode\U-2-Net\saved_models\u2netp\u2netp_epoch_300.pth"



# 4. 可视化配置
OVERLAY_COLOR = (0, 0, 255)  # 红色 (BGR格式)
MAX_ALPHA = 0.7  # 最大透明度

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
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
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


def apply_clahe(image):
    """CLAHE 增强 (与训练代码一致)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image
    
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_new = cv2.merge((l_clahe, a, b))
    img_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    
    return img_new


def preprocess_image(image_path, input_size, use_clahe=True):
    """预处理流水线"""
    img_bgr = cv2_read_img(image_path)
    if img_bgr is None:
        return None, None, None

    # 灰度图转3通道
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    original_shape = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(
        img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR
    )

    # CLAHE
    if use_clahe:
        img_resized = apply_clahe(img_resized)

    # 归一化 + 标准化
    img_norm = img_resized.astype(np.float32) / 255.0
    tmpImg = np.zeros_like(img_norm)
    tmpImg[:, :, 0] = (img_norm[:, :, 0] - 0.485) / 0.229
    tmpImg[:, :, 1] = (img_norm[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (img_norm[:, :, 2] - 0.406) / 0.225

    # HWC -> CHW -> NCHW
    tmpImg = tmpImg.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(tmpImg).unsqueeze(0).float()

    return img_tensor, original_shape, img_bgr


def load_model(config):
    """根据配置加载模型"""
    model_name = config['model_name']
    
    # 确定模型文件
    if MODEL_FILE:
        model_file = MODEL_FILE
    else:
        model_file = config.get('best_model', f"{model_name}_epoch_{config['total_epochs']}.pth")
    
    model_path = os.path.join(MODEL_DIR, model_file)
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    logger.info(f"加载模型: {model_path}")
    
    # 实例化网络
    if model_name == "u2net":
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        net = U2NETP(3, 1)
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


def predict(net, img_tensor):
    """模型推理"""
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        d0, *_ = net(img_tensor)
        pred = d0[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    return pred.cpu().numpy().squeeze()


def draw_labelme_annotations(img, json_path):
    """在图像上绘制 labelme 标注 (如果存在)"""
    if not os.path.exists(json_path):
        return img
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return img
    
    img_draw = img.copy()
    ANNOTATION_COLOR = (0, 255, 0)  # 绿色
    LINE_THICKNESS = 2
    
    for shape in data.get('shapes', []):
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])
        
        if not points:
            continue
        
        pts = np.array(points, dtype=np.int32)
        
        if shape_type == 'polygon':
            cv2.polylines(img_draw, [pts], isClosed=True, color=ANNOTATION_COLOR, thickness=LINE_THICKNESS)
        elif shape_type == 'rectangle' and len(pts) >= 2:
            cv2.rectangle(img_draw, tuple(pts[0]), tuple(pts[1]), color=ANNOTATION_COLOR, thickness=LINE_THICKNESS)
        elif shape_type == 'circle' and len(pts) >= 2:
            center = tuple(pts[0])
            radius = int(np.linalg.norm(pts[0] - pts[1]))
            cv2.circle(img_draw, center, radius, color=ANNOTATION_COLOR, thickness=LINE_THICKNESS)
    
    return img_draw


def overlay_result(original_img, pred_mask, output_path, img_path=None):
    """生成可视化结果"""
    h, w = original_img.shape[:2]
    
    # 还原 mask 到原图尺寸
    mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 热力图叠加
    heatmap = np.zeros_like(original_img)
    heatmap[:] = OVERLAY_COLOR
    
    alpha = mask_resized * MAX_ALPHA
    alpha[alpha < 0.1] = 0
    alpha = np.stack([alpha] * 3, axis=-1)
    
    overlay = original_img * (1 - alpha) + heatmap * alpha
    overlay = overlay.astype(np.uint8)
    
    # 原图 (带标注)
    original_with_annotation = original_img.copy()
    if img_path:
        base_path = os.path.splitext(img_path)[0]
        json_path = base_path + '.json'
        original_with_annotation = draw_labelme_annotations(original_with_annotation, json_path)
    
    # 拼接: 上=预测结果, 下=原图(带GT标注)
    combined = np.vstack([overlay, original_with_annotation])
    cv2.imwrite(output_path, combined)


# =====================================================================
#                           主函数
# =====================================================================

def main():
    # 1. 加载配置
    config = load_config()
    if config is None:
        return
    
    input_size = tuple(config['input_size'])
    use_clahe = config.get('use_clahe', True)
    
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
        img_tensor, orig_shape, orig_img_bgr = preprocess_image(img_path, input_size, use_clahe)
        if img_tensor is None:
            logger.warning(f"无法读取: {fname}")
            continue
        
        # 推理
        t_start = time.perf_counter()
        pred_mask = predict(net, img_tensor)
        t_end = time.perf_counter()
        
        infer_time = (t_end - t_start) * 1000
        total_time += infer_time
        
        # 保存结果
        save_path = os.path.join(OUTPUT_DIR, fname)
        overlay_result(orig_img_bgr, pred_mask, save_path, img_path)
        
        logger.info(f"[{i+1}/{len(image_list)}] {fname} - {infer_time:.1f}ms")
    
    logger.info("")
    logger.success(f"推理完成! 平均耗时: {total_time/len(image_list):.1f}ms")
    logger.info(f"结果保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
