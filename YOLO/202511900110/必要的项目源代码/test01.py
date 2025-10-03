import cv2
import os
import argparse
import time
from ultralytics import YOLO
from pathlib import Path

def process_single_image(model, img_path, save_dir, conf_threshold=0.5):
    """处理单张图片并保存结果"""
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图片 {img_path}")
        return
    
    # 推理（强制CPU）
    results = model(img, conf=conf_threshold, device="cpu", verbose=False)
    
    # 绘制检测框
    annotated_img = results[0].plot(conf=True, labels=True)
    
    # 保存结果
    save_path = os.path.join(save_dir, "images", os.path.basename(img_path))
    cv2.imwrite(save_path, annotated_img)
    print(f"图片结果已保存：{save_path}")

def process_single_video(model, video_path, save_dir, conf_threshold=0.5):
    """处理单个视频并保存结果"""
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建保存视频的目录和文件
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, "videos", f"{video_name}_result.mp4")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_count = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame, conf=conf_threshold, device="cpu", verbose=False)
        annotated_frame = results[0].plot(conf=True, labels=True)
        
        # 写入结果视频
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"视频 {video_name} 处理中... 已处理 {frame_count} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    elapsed = time.time() - start_time
    print(f"视频结果已保存：{save_path}（耗时 {elapsed:.2f} 秒）")

def process_testset(model_path, testset_dir, conf_threshold=0.5):
    """批量处理测试集（图片和视频）"""
    # 1. 初始化模型
    print(f"加载模型：{model_path}")
    model = YOLO(model_path)
    
    # 2. 创建结果保存目录
    save_root = os.path.join(os.path.dirname(model_path), "test_results")
    os.makedirs(os.path.join(save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "videos"), exist_ok=True)
    print(f"所有结果将保存到：{save_root}")
    
    # 3. 遍历测试集目录，区分图片和视频
    supported_img_ext = (".jpg", ".jpeg", ".png", ".bmp")
    supported_video_ext = (".mp4", ".avi", ".mov", ".mkv")
    
    for file in os.listdir(testset_dir):
        file_path = os.path.join(testset_dir, file)
        if file.lower().endswith(supported_img_ext):
            # 处理图片
            process_single_image(model, file_path, save_root, conf_threshold)
        elif file.lower().endswith(supported_video_ext):
            # 处理视频
            process_single_video(model, file_path, save_root, conf_threshold)
        else:
            print(f"跳过不支持的文件：{file}")
    
    print("测试集处理完成！所有结果已保存。")

def run_camera_inference(model_path, conf_threshold=0.5):
    """摄像头实时推理（方便快速验证）"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头
    window_name = "摄像头实时推理"
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头推理启动，按 'q' 键退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame, conf=conf_threshold, device="cpu", verbose=False)
        annotated_frame = results[0].plot(conf=True, labels=True)
        
        # 显示
        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8测试集推理工具")
    parser.add_argument("--model", type=str, 
                        default="D:/yolo model new 2/yolo/ultralytics/runs/plate_detection/yolov8n_cpu_train/weights/best.pt",
                        help="模型权重路径（best.pt）")
    parser.add_argument("--testset", type=str, 
                        default="D:/yolo model new 2/testdatasets/images",  # 测试集目录
                        help="测试集文件夹路径")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="置信度阈值（推荐0.2-0.5）")
    parser.add_argument("--camera", action="store_true", 
                        help="使用摄像头实时推理（不处理测试集）")
    
    args = parser.parse_args()
    
    if args.camera:
        # 摄像头实时推理
        run_camera_inference(args.model, args.conf)
    else:
        # 批量处理测试集
        process_testset(args.model, args.testset, args.conf)
    