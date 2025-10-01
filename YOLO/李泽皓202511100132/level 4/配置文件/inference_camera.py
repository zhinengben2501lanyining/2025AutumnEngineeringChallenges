# inference_camera.py
import cv2
from ultralytics import YOLO
import time

def detect_camera(model, camera_id=0, conf_threshold=0.25):
    """
    使用摄像头进行实时车牌检测
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return
    
    print("📹 摄像头已启动，按 'q' 键退出，按 's' 键保存当前帧")
    
    frame_count = 0
    fps = 0
    start_time = time.time()
    
    # 创建保存目录
    import os
    save_dir = "captured_frames"
    os.makedirs(save_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头帧")
            break
        
        # 计算FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
        
        # 进行推理
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )
        
        # 处理结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 显示检测信息
        detection_info = f"检测到: {len(result.boxes)} 个车牌"
        fps_info = f"FPS: {fps:.1f}"
        
        # 在画面上添加文字信息
        cv2.putText(annotated_frame, detection_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, fps_info, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示实时画面
        cv2.imshow('Real-time License Plate Detection', annotated_frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按Q退出
            break
        elif key == ord('s'):  # 按S保存当前帧
            timestamp = int(time.time())
            save_path = f"{save_dir}/capture_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 帧已保存: {save_path}")
        elif key == ord('d'):  # 按D显示/隐藏检测信息
            # 可以添加切换显示检测信息的逻辑
            pass
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头检测已停止")

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "runs/detect/license_plate_detection_v1/weights/best.pt"
    model = load_model(model_path)
    
    # 开始实时检测
    detect_camera(model, camera_id=0)  # 0表示默认摄像头