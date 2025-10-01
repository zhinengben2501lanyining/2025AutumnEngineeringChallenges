# inference_main.py
"""
车牌检测推理主程序
支持图片、视频、摄像头实时检测
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLO车牌检测推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--source', type=str, required=True, help='输入源（图片/视频路径或摄像头ID）')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], help='检测模式')
    
    args = parser.parse_args()
    
    # 加载模型
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"✅ 模型加载成功: {args.model}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 自动检测模式
    if args.mode is None:
        if args.source.isdigit():
            args.mode = 'camera'
        elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            args.mode = 'image'
        else:
            args.mode = 'video'
    
    print(f"🎯 检测模式: {args.mode}")
    print(f"📁 输入源: {args.source}")
    
    # 执行检测
    if args.mode == 'image':
        from inference_image import detect_single_image
        detect_single_image(model, args.source, args.output, args.conf)
    
    elif args.mode == 'video':
        from inference_video import detect_video
        detect_video(model, args.source, args.output, args.conf)
    
    elif args.mode == 'camera':
        from inference_camera import detect_camera
        detect_camera(model, int(args.source), args.conf)

if __name__ == "__main__":
    main()