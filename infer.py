"""
U2Net 推理脚本 - 将预测掩码的轮廓叠加到原图上
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET


def cv2_read_img(file_path):
    """
    读图函数
    :param file_path: 图片路径
    :return: 图片变量-array
    """
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img
# ======= 用户配置 =======
# 测试图片目录
IMAGE_DIR = r'\\192.168.1.55\ai研究院\5_临时文件夹\czj\1.datatest\4_濠玮b402-刀纹\2_Skolpha\2_test\1_刀纹'
# 输出结果目录
OUTPUT_DIR = os.path.join(os.getcwd(), 'test_data', 'overlay_results')
# 训练好的模型路径
MODEL_PATH = r'F:\New_SourceCode\U-2-Net\saved_models\u2net\u2net_bce_itr_25000_train_0.992337_tar_0.060030.pth'

# 叠加颜色 (BGR格式)
OVERLAY_COLOR = (0, 0, 255)  # 绿色
MAX_ALPHA = 0.7  # 最大叠加透明度 (mask=1时的透明度)
# =======================


def load_model(model_path):
    """加载模型"""
    print(f"Loading model from: {model_path}")
    net = U2NET(3, 1)
    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    net.eval()
    print("Model loaded successfully!")
    return net


def preprocess_image(image_path):
    """
    预处理图像 - 与 ToTensorLab(flag=0) 保持一致
    """
    image = io.imread(image_path)
    original_shape = image.shape[:2]  # (H, W)
    
    # 转换为3通道
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Resize到320x320 - 关键：preserve_range=True 保持原始数据范围
    img_resized = transform.resize(image, (320, 320), mode='constant', preserve_range=True)
    
    # 归一化 - 与 ToTensorLab(flag=0) 一致
    # 1. 先用 np.max(image) 归一化到 0-1
    img_resized = img_resized / np.max(img_resized) if np.max(img_resized) > 0 else img_resized
    
    # 2. 然后用 ImageNet 均值和标准差
    img_normalized = np.zeros((320, 320, 3))
    img_normalized[:, :, 0] = (img_resized[:, :, 0] - 0.485) / 0.229
    img_normalized[:, :, 1] = (img_resized[:, :, 1] - 0.456) / 0.224
    img_normalized[:, :, 2] = (img_resized[:, :, 2] - 0.406) / 0.225
    
    # 转换为tensor
    img_tensor = img_normalized.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()
    
    return img_tensor, original_shape


def predict(net, img_tensor):
    """模型推理"""
    if torch.cuda.is_available():
        img_tensor = Variable(img_tensor.cuda())
    else:
        img_tensor = Variable(img_tensor)
    
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(img_tensor)
    
    # 取第一个输出（最终融合结果）
    pred = d1[:, 0, :, :]
    
    # 归一化到0-1
    pred = pred.squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    return pred.cpu().numpy()


def overlay_heatmap(original_image, mask, color=(0, 255, 0), max_alpha=0.7):
    """
    将模型输出的热力图直接叠加到原图上（灰色区域也会显示）
    
    Args:
        original_image: 原始图像 (H, W, 3) BGR格式
        mask: 预测掩码 (H, W) 0-1范围，值越大表示越可能是目标
        color: 叠加颜色 (B, G, R)
        max_alpha: 最大叠加透明度 (当mask=1时的透明度)
    
    Returns:
        overlay_image: 叠加了热力图的图像
        mean_confidence: 平均置信度
    """
    # 确保mask尺寸与原图一致
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # 复制原图
    overlay = original_image.copy().astype(np.float32)
    
    # 创建彩色叠加层
    color_layer = np.zeros_like(original_image, dtype=np.float32)
    color_layer[:] = color
    
    # 根据 mask 值计算每个像素的透明度 (mask值越大，叠加越明显)
    alpha_map = mask * max_alpha  # shape: (H, W)
    alpha_3ch = np.stack([alpha_map] * 3, axis=-1)  # shape: (H, W, 3)
    
    # 混合: overlay = original * (1 - alpha) + color * alpha
    overlay = overlay * (1 - alpha_3ch) + color_layer * alpha_3ch
    
    # 计算平均置信度
    mean_confidence = np.mean(mask)
    
    return overlay.astype(np.uint8), mean_confidence


def process_single_image(net, image_path, output_dir):
    """处理单张图片"""
    # 读取原图
    original = cv2_read_img(image_path)
    if original is None:
        print(f"Warning: Cannot read {image_path}")
        return
    
    # 预处理
    img_tensor, original_shape = preprocess_image(image_path)
    
    # 推理
    mask = predict(net, img_tensor)
    
    # 叠加热力图
    overlay, mean_conf = overlay_heatmap(
        original, mask,
        color=OVERLAY_COLOR,
        max_alpha=MAX_ALPHA
    )
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_overlay.jpg")
    
    cv2.imwrite(output_path, overlay)
    print(f"Processed: {filename} -> Mean confidence: {mean_conf:.4f}")
    
    return overlay, mean_conf


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    net = load_model(MODEL_PATH)
    
    # 获取所有图片
    import glob
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    print(f"Found {len(image_list)} images")
    
    # 处理每张图片
    total_conf = 0
    count = 0
    for image_path in image_list:
        result = process_single_image(net, image_path, OUTPUT_DIR)
        if result:
            _, conf = result
            total_conf += conf
            count += 1
    
    print(f"\nDone! Results saved to: {OUTPUT_DIR}")
    if count > 0:
        print(f"Average confidence: {total_conf / count:.4f}")


if __name__ == "__main__":
    main()
