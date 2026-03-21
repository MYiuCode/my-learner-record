脑肿瘤智能检测可视化系统

项目介绍

基于 YOLOv8 与 PySide6 开发的脑肿瘤医学影像智能检测工具，支持原图 / 结果图实时对比、检测框自定义、局部放大查看、数据统计与导出，适用于脑肿瘤 MRI 影像的快速标注与辅助诊断。

功能清单

核心功能

✅ 脑肿瘤目标自动检测（仅框选病变区域，不冗余标注）

✅ 左右分屏对比显示：左侧检测前原图 + 右侧检测后标注图

✅ 检测框样式自定义：方形 / 圆形切换、大小缩放、颜色切换

✅ 图片滚轮局部放大 + 右下角缩略图定位（已移除放大镜图标）

✅ 右侧实时统计面板：模型信息、检测数量、类别分布、置信度范围

✅ 支持单张图片、批量图片、视频检测

✅ 标注图保存 + 检测结果 JSON/CSV 导出

界面样式

暖杏色主题 + 纯黑字体
浅紫圆角按钮，布局清爽无重叠
中间面板最大化展示

检测前 / 检测后标题放大加粗 + 粗花环线框包裹

左右区域可随缩放移动的虚线分隔



环境依赖
txt

Python 3.8+

ultralytics >= 8.0

PySide6

opencv-python

numpy

一键安装：

bash

运行

pip install ultralytics pyside6 opencv-python numpy

模型使用说明

训练模型（已提供脚本）

运行 train_brain_tumor.py 自动下载数据集并训练：

python

运行

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="brain-tumor.yaml", epochs=30, imgsz=640, device="cpu")

模型路径

训练完成后模型位于：

runs/detect/train/weights/best.pt

加载模型

软件内点击 加载模型权重 → 选择 best.pt 即可使用。

文件结构 plaintext

brain-tumor-detector/

├── main.py                # 主程序（完整GUI）

├── train_brain_tumor.py   # 模型训练脚本

├── best.pt                # 训练好的脑肿瘤权重

├── README.md              # 说明文档

└── runs/                  # 训练输出目录

注意事项

模型需先训练或使用提供的权重，不可直接运行空模型
支持格式：jpg /jpeg/png /bmp/mp4 /avi
放大查看时缩略图会自动标记当前视野区域
检测结果默认保存置信度 ≥ 0.25 的目标