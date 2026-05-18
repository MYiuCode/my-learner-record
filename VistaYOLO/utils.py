from ultralytics import YOLO
import cv2

def detect_image(image_path, save_path="results/result.jpg"):
    # 加载YOLOv8模型
    model = YOLO("yolov8n.pt")
    # 识别图片
    results = model(image_path)
    # 读取图片并画框
    res_img = results[0].plot()
    # 保存结果到指定路径
    cv2.imwrite(save_path, res_img)
    return save_path