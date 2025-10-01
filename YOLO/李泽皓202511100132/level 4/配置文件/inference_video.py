# inference_video.py
import cv2
from ultralytics import YOLO
import os

def detect_video(model, video_path, output_dir="outputs", conf_threshold=0.25):
    """
    对视频文件进行车牌检测
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎥 视频信息: {fps}FPS, 分辨率: {width}x{height}, 总帧数: {total_frames}")
    
    # 创建视频写入器
    output_path = os.path.join(output_dir, "detected_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    print("⏳ 开始处理视频...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行推理
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False  # 不显示进度信息
        )
        
        # 处理当前帧的结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 统计检测结果
        if len(result.boxes) > 0:
            detection_count += 1
        
        # 写入输出视频
        out.write(annotated_frame)
        
        # 显示实时预览（可选）
        cv2.imshow('Video Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按Q键退出
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            progress = (frame_count / total_frames) * 100
            print(f"📊 处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ 视频处理完成!")
    print(f"📊 统计信息:")
    print(f"   - 总帧数: {frame_count}")
    print(f"   - 检测到车牌的帧数: {detection_count}")
    print(f"   - 检测率: {(detection_count/frame_count)*100:.1f}%")
    print(f"   - 输出文件: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "runs/detect/license_plate_detection_v1/weights/best.pt"
    model = load_model(model_path)
    
    # 检测视频文件
    video_path = "test_videos/traffic.mp4"  # 请替换为你的视频路径
    detect_video(model, video_path)