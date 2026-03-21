# my-learner-record

#我的学习记录
VistaYOLO 
基于 YOLOv8 的目标检测项目

项目简介
这是一个使用 Ultralytics YOLOv8 模型实现的图像目标检测项目，可快速加载预训练模型，对输入图片进行目标识别，并自动保存带检测框的结果图片。
 
功能特性
- 支持单张图片目标检测
- 自动保存检测结果到指定路径
- 基于 YOLOv8n 轻量模型，推理速度快
- 代码结构清晰，便于二次开发与扩展
 
安装依赖
bash
pip install -r requirements.txt

快速开始
1. 将待检测图片放入项目目录
2. 在代码中调用检测函数：
python
from utils import detect_image
# 传入图片路径，可自定义结果保存路径
detect_image("test.jpg", save_path="results/result.jpg") 
3. 运行后，检测结果将保存在  results/  文件夹中 
项目结构
plaintext
  
VistaYOLO/
├── main.py          # 主程序入口
├── utils.py         # 核心检测函数实现
├── requirements.txt # 项目依赖列表
├── yolov8n.pt       # YOLOv8 预训练模型文件
├── README.md        # 项目说明文档
└── results/         # 检测结果保存目录
    └── result.jpg   # 示例检测结果 
 
依赖说明
-  ultralytics : YOLOv8 官方库
-  opencv-python : 用于图片读写与可视化
- 其他依赖见  requirements.txt 
