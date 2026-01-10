import os
from datetime import datetime

# =====================================================================
#                           用户配置区域
# =====================================================================

# 1. 项目名称 (所有数据和模型都保存在 projects/<PROJECT_NAME>/ 下)
PROJECT_NAME = "xmy2"

# 2. 源数据路径 (labelme 格式: 图片 + 同名 JSON 文件)
#    支持递归搜索子文件夹
SOURCE_DATA_DIR = r"\\192.168.1.55\ai研究院\5_临时文件夹\czj\1.datatest\2_新美洋\2_Skolpha\1_train\100pcs"

# 3. 模型配置
MODEL_NAME = "u2netp"  # "u2netp" (轻量版) 或 "u2net" (完整版)

# 4. 训练超参数
# 输入尺寸配置 (MMDet 风格)
# 格式: (长边最大值, 短边目标值)
# 策略: 短边优先缩放到目标值，但长边不能超过最大值
INPUT_SCALE = (1024, 256)  # 如果只传 int，等价于正方形 (size, size)
BATCH_SIZE = 4
EPOCH_NUM = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# 学习率调度器配置
USE_COSINE_ANNEALING = False # 是否开启 CosineAnnealingWarmRestarts
COSINE_T0 = 20
COSINE_TMULT = 2

# 5. 多类别配置
#    - NUM_CLASSES = 1: 二值分割 (背景 vs 目标)，输出概率图 [0,1]
#    - NUM_CLASSES > 1: 多类别分割，输出每个类别的概率
#    - CLASS_NAMES: labelme 中的类别名称到类别索引的映射
#      注意: 背景固定为 0，其他类别从 1 开始编号
#      所以 NUM_CLASSES = len(CLASS_NAMES) + 1 (背景)
NUM_CLASSES = 6  # 5个缺陷类别 + 1个背景 = 6

# 类别名称映射 (仅 NUM_CLASSES > 1 时使用)
# 例如: {"scratch": 1, "stain": 2, "crack": 3}
# 背景自动为 0，不需要配置
CLASS_NAMES = {"Hxian": 1, "Mpo": 2, "Zwu": 3, "Qpao": 4, "Yshang": 5}
# CLASS_NAMES = {"maosi": 1, "yiwu": 2}
# CLASS_NAMES = {}

# 类别权重 (可选，用于处理类别不平衡)
# 例如: [1.0, 2.0, 3.0] 表示类别0权重1，类别1权重2...
CLASS_WEIGHTS = None

# 6. GPU 配置
USE_MULTI_GPU = False  # True: 使用所有可见GPU多卡并行训练; False: 仅使用单卡
CUDA_VISIBLE_DEVICES = "0" # 设置可见的 GPU 设备

# 7. 其他配置
NUM_WORKERS = 4  # DataLoader 工作线程数
SAVE_EVERY_N_EPOCHS = 20  # 每隔多少 epoch 保存一次模型

# =====================================================================
#                           自动计算路径
# =====================================================================

# 设置环境变量 (必须在 import torch 之前)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# 项目根目录
PROJECT_DIR = os.path.join(os.getcwd(), "projects", PROJECT_NAME)

# 转换后的数据目录
CONVERTED_DATA_DIR = os.path.join(PROJECT_DIR, "train_data")
IMAGES_DIR = os.path.join(CONVERTED_DATA_DIR, "images")
MASKS_DIR = os.path.join(CONVERTED_DATA_DIR, "masks")

# 模型保存目录
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "models")

# TensorBoard 日志目录
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_DIR, "runs")

# 预训练权重路径
PRETRAINED_PATH = os.path.join(os.getcwd(), "pretrain", f"{MODEL_NAME}.pth")
