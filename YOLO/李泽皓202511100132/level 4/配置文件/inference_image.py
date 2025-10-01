# inference_image.py
from ultralytics import YOLO
import cv2
import os

def detect_single_image(model, image_path, output_dir="outputs", conf_threshold=0.25):
    """
    对单张图片进行车牌检测
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return None
    
    print(f"🔍 正在检测图片: {image_path}")
    
    # 进行推理
    results = model.predict(
        source=image_path,
        conf=conf_threshold,  # 置信度阈值
        save=True,           # 保存结果图片
        project=output_dir,  # 输出目录
        name="image_results", # 子目录名称
        exist_ok=True        # 覆盖已存在的目录
    )
    
    # 处理结果
    for i, result in enumerate(results):
        # 获取原始图像（带检测框）
        annotated_image = result.plot()  # 这个图像已经画好了检测框
        
        # 显示检测信息
        print(f"📷 图片 {i+1}:")
        if len(result.boxes) > 0:
            for j, box in enumerate(result.boxes):
                confidence = box.conf.item()  # 置信度
                class_id = int(box.cls.item())  # 类别ID
                class_name = model.names[class_id]  # 类别名称
                
                print(f"   🚗 检测到 {class_name}: 置信度 {confidence:.3f}")
                
                # 获取边界框坐标（像素坐标）
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"     位置: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
        else:
            print("   ❌ 未检测到车牌")
        
        # 显示图片（可选）
        cv2.imshow('Detection Result', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "runs/detect/license_plate_detection_v1/weights/best.pt"
    model = load_model(model_path)
    
    # 检测单张图片
    image_path = "test_images/car1.jpg"  # 请替换为你的测试图片路径
    results = detect_single_image(model, image_path)