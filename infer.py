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
MODEL_PATH = r'F:\New_SourceCode\U-2-Net\saved_models\u2net\u2net_bce_itr_156_train_3.483751_tar_0.451071.pth'

# 轮廓颜色 (BGR格式)
CONTOUR_COLOR = (0, 255, 0)  # 绿色
CONTOUR_THICKNESS = 2  # 轮廓线宽
MASK_THRESHOLD = 0.5  # 掩码二值化阈值
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
    """预处理图像"""
    image = io.imread(image_path)
    original_shape = image.shape[:2]  # (H, W)
    
    # 转换为3通道
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Resize到320x320
    img_resized = transform.resize(image, (320, 320), mode='constant')
    
    # 归一化
    img_normalized = np.zeros((320, 320, 3))
    img_resized = img_resized / np.max(img_resized) if np.max(img_resized) > 0 else img_resized
    
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


def overlay_contour(original_image, mask, threshold=0.5, color=(0, 255, 0), thickness=2):
    """
    将掩码轮廓叠加到原图上
    
    Args:
        original_image: 原始图像 (H, W, 3) BGR格式
        mask: 预测掩码 (H, W) 0-1范围
        threshold: 二值化阈值
        color: 轮廓颜色 (B, G, R)
        thickness: 轮廓线宽
    
    Returns:
        overlay_image: 叠加了轮廓的图像
    """
    # 确保mask尺寸与原图一致
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # 二值化
    mask_binary = (mask > threshold).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制原图
    overlay = original_image.copy()
    
    # 绘制轮廓
    cv2.drawContours(overlay, contours, -1, color, thickness)
    
    return overlay, len(contours)


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
    
    # 叠加轮廓
    overlay, num_contours = overlay_contour(
        original, mask,
        threshold=MASK_THRESHOLD,
        color=CONTOUR_COLOR,
        thickness=CONTOUR_THICKNESS
    )
    
    # 保存结果
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_overlay.jpg")
    
    cv2.imwrite(output_path, overlay)
    print(f"Processed: {filename} -> Found {num_contours} contours")
    
    return overlay, num_contours


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
    total_contours = 0
    for image_path in image_list:
        _, num = process_single_image(net, image_path, OUTPUT_DIR)
        if num:
            total_contours += num
    
    print(f"\nDone! Results saved to: {OUTPUT_DIR}")
    print(f"Total contours found: {total_contours}")


if __name__ == "__main__":
    main()
