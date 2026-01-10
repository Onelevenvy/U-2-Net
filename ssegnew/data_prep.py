import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
import glob
from loguru import logger

import config as cfg

def polygon_to_mask(points, image_width, image_height):
    """将多边形点列表转换为二值mask"""
    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    polygon_points = [(int(p[0]), int(p[1])) for p in points]
    draw.polygon(polygon_points, fill=255)
    return np.array(mask)


def convert_labelme_json(json_path, output_mask_path, num_classes=1, class_names=None):
    """
    将单个labelme JSON文件转换为mask图片

    Args:
        json_path: labelme JSON 文件路径
        output_mask_path: 输出 mask 路径
        num_classes: 类别数
            - 1: 二值分割，所有标注合并为单一前景 (值=255)
            - >1: 多类别分割，根据类别名称分配类别索引
        class_names: 类别名称到索引的映射 {"label_name": class_idx}
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        # 创建空白mask
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # 遍历所有shapes
        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type", "")
            points = shape.get("points", [])
            label_name = shape.get("label", "unknown")

            # 确定像素值
            if num_classes == 1:
                # 二值分割: 所有前景都是 255
                pixel_value = 255
            else:
                # 多类别分割: 根据类别名称获取索引
                if class_names and label_name in class_names:
                    pixel_value = class_names[label_name]
                else:
                    # 未知类别，跳过或者使用默认值
                    logger.warning(f"未知类别 '{label_name}'，跳过")
                    continue

            # 绘制形状
            if shape_type == "polygon" and len(points) >= 3:
                mask = polygon_to_mask(points, image_width, image_height)
                # 对于多类别，后绘制的覆盖前面的
                combined_mask[mask > 0] = pixel_value

            elif shape_type == "rectangle" and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                combined_mask[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)] = (
                    pixel_value
                )

            elif shape_type == "circle" and len(points) == 2:
                cx, cy = int(points[0][0]), int(points[0][1])
                ex, ey = int(points[1][0]), int(points[1][1])
                radius = int(np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2))
                cv2.circle(combined_mask, (cx, cy), radius, int(pixel_value), -1)

        # 保存mask
        mask_img = Image.fromarray(combined_mask)
        mask_img.save(output_mask_path)
        return True

    except Exception as e:
        logger.error(f"转换失败 {json_path}: {e}")
        return False


def convert_dataset():
    """
    自动将源数据转换为 U2Net 格式
    返回: 是否需要重新转换 (True=已转换, False=失败)
    """
    # 检查是否已经转换过
    if os.path.exists(cfg.IMAGES_DIR) and os.path.exists(cfg.MASKS_DIR):
        existing_images = len(glob.glob(os.path.join(cfg.IMAGES_DIR, "*.*")))
        existing_masks = len(glob.glob(os.path.join(cfg.MASKS_DIR, "*.png")))

        if existing_images > 0 and existing_masks > 0:
            logger.info(
                f"检测到已转换的数据: {existing_images} 张图片, {existing_masks} 个mask"
            )
            logger.info("跳过转换步骤，直接使用已有数据")
            return True

    logger.info("=" * 50)
    logger.info("开始转换 Labelme 数据...")
    logger.info(f"源数据目录: {cfg.SOURCE_DATA_DIR}")
    logger.info(f"输出目录: {cfg.CONVERTED_DATA_DIR}")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(cfg.IMAGES_DIR, exist_ok=True)
    os.makedirs(cfg.MASKS_DIR, exist_ok=True)

    input_path = Path(cfg.SOURCE_DATA_DIR)

    # 递归查找所有JSON文件
    json_files = list(input_path.rglob("*.json"))

    if len(json_files) == 0:
        logger.error(f"未找到 JSON 文件！请检查源数据目录: {cfg.SOURCE_DATA_DIR}")
        return False

    logger.info(f"找到 {len(json_files)} 个 JSON 文件")

    success_count = 0
    fail_count = 0

    for json_file in json_files:
        base_name = json_file.stem
        json_dir = json_file.parent

        # 尝试多种图片扩展名
        image_file = None
        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
        ]:
            possible_image = json_dir / f"{base_name}{ext}"
            if possible_image.exists():
                image_file = possible_image
                break

        if image_file is None:
            logger.warning(f"找不到对应的图片: {base_name}")
            fail_count += 1
            continue

        # 输出路径
        output_image = Path(cfg.IMAGES_DIR) / f"{base_name}{image_file.suffix}"
        output_mask = Path(cfg.MASKS_DIR) / f"{base_name}.png"

        # 复制图片
        shutil.copy2(image_file, output_image)

        # 转换JSON为mask
        if convert_labelme_json(json_file, output_mask, cfg.NUM_CLASSES, cfg.CLASS_NAMES):
            success_count += 1
        else:
            fail_count += 1

    logger.info("=" * 50)
    logger.info(f"转换完成! 成功: {success_count}, 失败: {fail_count}")
    if cfg.NUM_CLASSES > 1:
        logger.info(f"多类别模式: {cfg.NUM_CLASSES} 个类别")
        logger.info(f"类别映射: {cfg.CLASS_NAMES}")
    logger.info("=" * 50)

    return success_count > 0
