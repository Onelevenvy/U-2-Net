"""
Labelme JSON 转 U-2-Net 数据集格式

用法:
    python convert_labelme_to_u2net.py --input_dir <labelme数据目录> --output_dir <输出目录>

输入目录结构:
    input_dir/
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    ├── image2.json
    └── ...

输出目录结构:
    output_dir/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── masks/
        ├── image1.png
        ├── image2.png
        └── ...
"""

import os
import json
import argparse
import shutil
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path


def polygon_to_mask(points, image_width, image_height):
    """
    将多边形点列表转换为二值mask
    
    Args:
        points: 多边形顶点列表 [[x1,y1], [x2,y2], ...]
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        numpy array: 二值mask (0和255)
    """
    # 创建空白mask
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 将点转换为元组列表
    polygon_points = [(int(p[0]), int(p[1])) for p in points]
    
    # 绘制填充的多边形
    draw.polygon(polygon_points, fill=255)
    
    return np.array(mask)


def convert_labelme_json(json_path, output_mask_path):
    """
    将单个labelme JSON文件转换为mask图片
    
    Args:
        json_path: labelme JSON文件路径
        output_mask_path: 输出mask图片路径
    
    Returns:
        bool: 是否成功
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 创建空白mask
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # 遍历所有shapes
        for shape in data.get('shapes', []):
            shape_type = shape.get('shape_type', '')
            points = shape.get('points', [])
            
            if shape_type == 'polygon' and len(points) >= 3:
                # 多边形标注
                mask = polygon_to_mask(points, image_width, image_height)
                combined_mask = np.maximum(combined_mask, mask)
            
            elif shape_type == 'rectangle' and len(points) == 2:
                # 矩形标注 (两个对角点)
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                combined_mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 255
            
            elif shape_type == 'circle' and len(points) == 2:
                # 圆形标注 (中心点和边缘点)
                cx, cy = int(points[0][0]), int(points[0][1])
                ex, ey = int(points[1][0]), int(points[1][1])
                radius = int(np.sqrt((ex-cx)**2 + (ey-cy)**2))
                cv2.circle(combined_mask, (cx, cy), radius, 255, -1)
        
        # 保存mask
        mask_img = Image.fromarray(combined_mask)
        mask_img.save(output_mask_path)
        
        return True
        
    except Exception as e:
        print(f"转换失败 {json_path}: {e}")
        return False


def convert_dataset(input_dir, output_dir, image_ext='.jpg', recursive=True):
    """
    批量转换整个数据集
    
    Args:
        input_dir: 输入目录 (包含图片和JSON文件)
        output_dir: 输出目录
        image_ext: 图片扩展名
        recursive: 是否递归搜索子文件夹
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件 (支持递归)
    if recursive:
        json_files = list(input_path.rglob('*.json'))  # rglob 递归搜索
    else:
        json_files = list(input_path.glob('*.json'))
    
    success_count = 0
    fail_count = 0
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        # 获取对应的图片文件名
        base_name = json_file.stem
        json_dir = json_file.parent  # JSON文件所在目录
        
        # 尝试多种图片扩展名 (在JSON文件同目录下查找)
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']:
            possible_image = json_dir / f"{base_name}{ext}"
            if possible_image.exists():
                image_file = possible_image
                break
        
        if image_file is None:
            print(f"警告: 找不到对应的图片文件 {base_name}")
            fail_count += 1
            continue
        
        # 输出路径
        output_image = images_dir / f"{base_name}{image_file.suffix}"
        output_mask = masks_dir / f"{base_name}.png"
        
        # 复制图片
        shutil.copy2(image_file, output_image)
        
        # 转换JSON为mask
        if convert_labelme_json(json_file, output_mask):
            success_count += 1
            print(f"✓ 转换成功: {base_name}")
        else:
            fail_count += 1
    
    print(f"\n转换完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"\n输出目录结构:")
    print(f"  {output_path}")
    print(f"  ├── images/  ({success_count} 张图片)")
    print(f"  └── masks/   ({success_count} 个mask)")
    
    # 生成训练配置提示
    print(f"\n使用方法:")
    print(f"在 u2net_train.py 中修改以下路径:")
    print(f"  data_dir = '{output_path}'")
    print(f"  tra_image_dir = 'images/'")
    print(f"  tra_label_dir = 'masks/'")


def main():
    parser = argparse.ArgumentParser(description='Labelme JSON 转 U-2-Net 数据集格式')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Labelme数据目录 (包含图片和JSON文件)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return
    
    convert_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
