import cv2
import argparse
import time
from ultralytics import YOLO  # 导入YOLOv8库

def yolov8_realtime_inference(model_path, source, conf_threshold=0.5):
    """
    YOLOv8 实时推理函数
    :param model_path: 训练好的模型路径（.pt文件）
    :param source: 输入源（摄像头编号如"0"，或视频文件路径如"test.mp4"）
    :param conf_threshold: 置信度阈值（过滤低置信度检测结果）
    """
    # 1. 加载训练好的模型（指定CPU设备，避免GPU兼容性问题）
    print(f"正在加载模型：{model_path}")
    model = YOLO(model_path)  # 加载权重文件
    print("模型加载完成，开始推理...")

    # 2. 打开输入源（摄像头或视频文件）
    # 判断输入源是摄像头（数字）还是视频文件（路径字符串）
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))  # 打开摄像头（0为默认摄像头）
        window_title = "YOLOv8 车牌检测（摄像头实时）"
    else:
        cap = cv2.VideoCapture(source)  # 打开视频文件
        window_title = f"YOLOv8 车牌检测（视频：{source}）"

    # 检查输入源是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开输入源 {source}")
        return

    # 3. 获取视频基本信息（宽度、高度、帧率）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"输入源信息：分辨率 {frame_width}x{frame_height}")

    # 4. 初始化FPS计算变量
    prev_time = time.time()  # 上一帧的时间
    fps = 0  # 实时帧率

    # 5. 循环读取帧并推理
    while cap.isOpened():
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("推理结束（视频已播放完毕或摄像头已断开）")
            break

        # ---------------------- 核心：YOLOv8推理 ----------------------
        # 强制用CPU推理（device="cpu"），避免GPU架构不兼容问题
        results = model(
            frame,
            conf=conf_threshold,  # 置信度阈值
            device="cpu",         # 关键：指定CPU设备
            verbose=False         # 关闭推理过程日志（减少输出干扰）
        )

        # ---------------------- 结果可视化 ----------------------
        # 在原图上绘制检测框、类别名称和置信度（YOLOv8内置plot()方法）
        annotated_frame = results[0].plot(
            conf=True,   # 显示置信度
            labels=True, # 显示类别名称
            boxes=True   # 显示检测框
        )

        # ---------------------- 计算并显示FPS ----------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)  # 计算当前帧率
        prev_time = curr_time

        # 在图像左上角绘制FPS（绿色文字，加粗）
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",  # 取整数FPS值
            (20, 50),           # 文字位置（x,y）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            1.2,                # 字体大小
            (0, 255, 0),        # 文字颜色（绿色）
            2                   # 文字粗细
        )

        # ---------------------- 显示结果 ----------------------
        cv2.imshow(window_title, annotated_frame)

        # 按下「q键」退出推理（等待1毫秒获取键盘输入）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户按下q键，退出推理")
            break

    # 6. 释放资源（关闭摄像头/视频文件，销毁窗口）
    cap.release()
    cv2.destroyAllWindows()
    print("资源已释放，程序结束")

# ---------------------- 命令行参数解析（方便快速切换输入源） ----------------------
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="YOLOv8 车牌检测实时推理程序")
    
    # 添加参数（默认值已适配你的模型路径和常见场景）
    parser.add_argument(
        "--model",
        type=str,
        default="D:/yolo model new 2/yolo/ultralytics/runs/plate_detection/yolov8n_cpu_train/weights/best.pt",
        help="训练好的模型路径（默认：你的best.pt路径）"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",  # 默认使用电脑摄像头（编号0）
        help="输入源（摄像头编号如'0'，或视频文件路径如'test_video.mp4'）"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default="0.5",  # 默认置信度阈值0.5（过滤低置信度结果）
        help="置信度阈值（如0.3、0.5，值越高误检越少）"
    )

    # 解析参数
    args = parser.parse_args()

    # 调用推理函数（传入解析后的参数）
    yolov8_realtime_inference(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf
    )