from ultralytics import YOLO

# 1. 加载YOLOv8n（轻量，速度快）
model = YOLO("yolov8n.pt")

# 2. 用官方脑肿瘤数据集自动训练（会自动下载数据集）
# 数据集地址：https://ultralytics.com/assets/brain-tumor.zip
model.train(
    data="brain-tumor.yaml",  # 官方配置，自动识别
    epochs=30,                # 训练轮数，30轮足够
    imgsz=640,
    batch=16,
    device="cpu"              # 有GPU就写0，没有就cpu
)

# 训练完，best.pt 会在 runs/detect/train/weights/ 里