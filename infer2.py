"""
U2NetP 推理脚本 - 适配工业质检 (长方形输入 + CLAHE增强)
"""

import os
import cv2
import torch
import numpy as np
from model import U2NETP  # 必须用 U2NETP

# ======= 用户配置 =======
# 1. 测试图片目录
IMAGE_DIR = r"F:\DL项目测试\4_濠玮b402-刀纹\2_Skolpha\2_test\1_刀纹"
# IMAGE_DIR = r"\\192.168.1.55\ai研究院\5_临时文件夹\czj\1.datatest\4_濠玮b402-刀纹\2_Skolpha\1_train\100pcs"

# 2. 训练好的模型路径 (确保是 u2netp 的权重)
MODEL_PATH = r"F:\New_SourceCode\U-2-Net\saved_models\u2netp\u2netp_epoch_200.pth"

# 3. 输出结果目录
OUTPUT_DIR = os.path.join(os.getcwd(), "test_results")

# 4. 输入尺寸 (Height, Width) - 必须与训练时的 input_size 一致！
# 原图是 2000x480，训练时建议等比例缩小到640以内
INPUT_SIZE = (224, 512)

# 5. 叠加显示配置
OVERLAY_COLOR = (0, 0, 255)  # 红色 (BGR格式)
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值 (高于此值才显示)
MAX_ALPHA = 0.7  # 最大透明度
# =======================


def cv2_read_img(file_path):
    """解决 Windows 下中文路径读取问题"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def apply_clahe(image):
    """
    CLAHE 增强 (与训练代码完全一致)
    image: numpy array (H, W, 3) RGB格式
    """
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 确保是 uint8
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image

    # 转到 LAB 空间
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 对 L 通道做 CLAHE
    l_clahe = clahe.apply(l)

    # 合并并转回 RGB
    lab_new = cv2.merge((l_clahe, a, b))
    img_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)

    return img_new


def preprocess_image(image_path, target_size):
    """
    预处理流水线：读取 -> 强制转3通道 -> Resize(长方形) -> CLAHE -> Normalize
    """
    # 1. 读取 (BGR)
    img_bgr = cv2_read_img(image_path)
    if img_bgr is None:
        return None, None, None

    # 【新增修复代码】如果是灰度图(2维)，强制转为3通道BGR
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    # 如果是4通道(带透明度)，去掉透明度
    elif img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    original_shape = img_bgr.shape[:2]  # (H, W)

    # 转 RGB (现在 img_bgr 肯定是3通道了，这行代码安全了)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Resize (注意 cv2.resize 接收 (Width, Height))
    img_resized = cv2.resize(
        img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )

    # 3. CLAHE 增强
    img_clahe = apply_clahe(img_resized)

    # 4. 归一化 (0-1)
    img_norm = img_clahe.astype(np.float32) / 255.0

    # 标准化 (ImageNet Mean/Std)
    tmpImg = np.zeros_like(img_norm)
    tmpImg[:, :, 0] = (img_norm[:, :, 0] - 0.485) / 0.229
    tmpImg[:, :, 1] = (img_norm[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (img_norm[:, :, 2] - 0.406) / 0.225

    # HWC -> CHW
    tmpImg = tmpImg.transpose((2, 0, 1))

    # 增加 Batch 维度: (1, 3, H, W)
    img_tensor = torch.from_numpy(tmpImg).unsqueeze(0).float()

    # 返回时，img_bgr 已经是3通道的了，overlay_result 就不会报错了
    return img_tensor, original_shape, img_bgr


def load_model(model_path):
    print(f"Loading U2NETP from: {model_path}")
    # 注意：这里实例化的是 U2NETP (Lite版)
    # 这里的 3 是 RGB通道，如果你训练时改成了 4 通道(加Sobel)，这里要改成 4
    net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))

    net.eval()
    return net


def predict(net, img_tensor):
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()  # 不需要 Variable，PyTorch 0.4+ 已弃用

    with torch.no_grad():
        # U2NETP 返回 tuple (d0, d1, ... d6)
        # d0 是主输出，d1-d6 是深监督输出（推理时不需要）
        d0, *_ = net(img_tensor)  # 忽略 d1-d6，更清晰

        # 输出是 Logits，需要 Sigmoid 转成概率
        pred = torch.sigmoid(d0[:, 0, :, :])

        # 归一化到 0-1，防止极端值影响可视化
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    return pred.cpu().numpy().squeeze()  # (H, W)


def overlay_result(original_img, pred_mask, output_path):
    """
    将预测结果叠加回原图
    """
    h, w = original_img.shape[:2]

    # 1. 将预测 mask (320, 1024) 还原回原图尺寸 (2000, 480)
    # 注意 cv2.resize 接收 (W, H)
    mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. 制作热力图层
    heatmap = np.zeros_like(original_img)
    heatmap[:] = OVERLAY_COLOR  # 填充颜色

    # 3. 制作 Alpha 通道
    # mask 值越大，透明度越高 (只显示高置信度区域)
    alpha = mask_resized * MAX_ALPHA

    # 简单的阈值过滤，让背景更干净
    alpha[alpha < 0.1] = 0

    alpha = np.stack([alpha] * 3, axis=-1)  # (H, W, 3)

    # 4. 融合
    # result = original * (1-alpha) + heatmap * alpha
    overlay = original_img * (1 - alpha) + heatmap * alpha
    overlay = overlay.astype(np.uint8)

    # 5. 另外保存一张纯 Mask 对比
    mask_uint8 = (mask_resized * 255).astype(np.uint8)

    # 拼接显示: 上面是原图叠加，下面是纯Mask
    combined = np.vstack([overlay, cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)])

    cv2.imwrite(output_path, combined)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载模型
    net = load_model(MODEL_PATH)

    # 2. 遍历图片
    import glob

    # 支持多种后缀
    exts = ["*.jpg", "*.png", "*.bmp"]
    image_list = []
    for ext in exts:
        image_list.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    print(f"Found {len(image_list)} images.")

    for i, img_path in enumerate(image_list):
        fname = os.path.basename(img_path)
        # print(f"[{i+1}/{len(image_list)}] Processing: {fname}")

        # 3. 预处理
        img_tensor, orig_shape, orig_img_bgr = preprocess_image(img_path, INPUT_SIZE)

        if img_tensor is None:
            continue

        # 4. 推理
        import time

        s = time.perf_counter()
        pred_mask = predict(net, img_tensor)
        e = time.perf_counter()

        t = (e - s) * 1000
        print("====================", t)

        # 5. 保存结果
        save_path = os.path.join(OUTPUT_DIR, fname)
        overlay_result(orig_img_bgr, pred_mask, save_path)

    print("Test finished!")


if __name__ == "__main__":
    main()
