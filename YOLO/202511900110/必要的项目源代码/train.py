from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 训练参数中强制使用 CPU
results = model.train(
    data="D:/yolo model new 2/yolo/datasets/CCPD/data.yaml",  # 你的数据集路径
    epochs=100,
    batch=2,  # CPU 训练批次不宜过大（避免内存溢出）
    imgsz=640,  # 减小图片尺寸，加快 CPU 处理速度
    device="cpu",  # 关键：强制使用 CPU
    verbose=True,  # 显示训练日志
    project="runs/plate_detection",
    name="yolov8n_cpu_train"
)
