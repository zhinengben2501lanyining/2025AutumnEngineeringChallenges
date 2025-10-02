import cv2
import os
from ultralytics import YOLO
import argparse

def detect_image(model, image_path, output_dir="output_images"):
    """对单张图片进行检测"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图片并检测
    results = model(image_path)
    
    # 保存检测结果
    for i, r in enumerate(results):
        # 绘制检测框
        im_array = r.plot()  # 生成带检测框的图像数组
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # 转换颜色通道
        
        # 保存结果
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, im)
        print(f"图片检测结果已保存至: {output_path}")
    
    # 显示结果（可选）
    cv2.imshow("Detection Result", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(model, video_path, output_dir="output_videos"):
    """对视频文件进行检测"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 定义输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, f"result_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("正在处理视频...（按 'q' 键退出）")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为RGB格式进行检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测
        results = model(rgb_frame)
        
        # 绘制检测框
        im_array = results[0].plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        
        # 写入输出视频
        out.write(im)
        
        # 显示实时处理结果
        cv2.imshow("Video Detection", im)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频检测结果已保存至: {output_path}")

def detect_camera(model):
    """使用摄像头进行实时检测"""
    # 打开默认摄像头（0表示默认摄像头，1表示外接摄像头）
    cap = cv2.VideoCapture(0)
    
    print("正在进行摄像头实时检测...（按 'q' 键退出）")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break
        
        # 检测
        results = model(frame)
        
        # 绘制检测框
        im_array = results[0].plot()
        
        # 显示结果
        cv2.imshow("Camera Real-time Detection", im_array)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="车牌检测脚本")
    parser.add_argument("--model", type=str, default="D:\CCPD2020\runs\detect\exp_merged_data8\weights\best.pt", 
                      help="模型权重文件路径")
    parser.add_argument("--mode", type=str, default="image", 
                      choices=["image", "video", "camera"], 
                      help="检测模式：image(图片), video(视频), camera(摄像头)")
    parser.add_argument("--path", type=str, default="D:\\CCPD2020\\test_set", 
                      help="图片或视频文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"正在加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 根据模式进行检测
    if args.mode == "image":
        # 如果是目录，则处理目录下所有图片
        if os.path.isdir(args.path):
            for file in os.listdir(args.path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    detect_image(model, os.path.join(args.path, file))
        else:
            # 处理单张图片
            detect_image(model, args.path)
    elif args.mode == "video":
        detect_video(model, args.path)
    elif args.mode == "camera":
        detect_camera(model)
    