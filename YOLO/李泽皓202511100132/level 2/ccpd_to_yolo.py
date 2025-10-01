import os
import cv2
import argparse

def parse_filename(filename):
    """
    解析CCPD文件名，提取边界框坐标
    :param filename: 
    :return: 如果解析成功，返回 x1, y1, x2, y2；否则返回 None
    """
    parts = filename.split('-')
    if len(parts) < 4:
        return None
        
    # 提取车牌区域坐标部分，例如 '154&383_386&473'
    bbox_str = parts[2]
    bbox_coords = bbox_str.split('_')
    if len(bbox_coords) < 2:
        return None
        
    # 解析左上角和右下角坐标
    left_top = bbox_coords[0].split('&')
    right_bottom = bbox_coords[1].split('&')
    
    if len(left_top) < 2 or len(right_bottom) < 2:
        return None
        
    try:
        x1 = int(left_top[0])
        y1 = int(left_top[1])
        x2 = int(right_bottom[0])
        y2 = int(right_bottom[1])
    except ValueError:
        return None
        
    return x1, y1, x2, y2

def convert_to_yolo_format(img_width, img_height, box):
    """
    将坐标转换为YOLO格式
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param box: 边界框坐标 (x1, y1, x2, y2)
    :return: YOLO格式的字符串
    """
    x1, y1, x2, y2 = box
    
    # 计算中心点和宽高
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # YOLO格式：class_id center_x center_y width height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_dataset(input_dir, output_dir):
    """
    处理整个数据集
    :param input_dir: 包含CCPD图片的输入目录
    :param output_dir: 输出目录
    """
    # 创建输出目录结构
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 统计处理情况
    processed_count = 0
    skipped_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {filename}")
            skipped_count += 1
            continue
            
        height, width = img.shape[:2]
        
        # 解析文件名中的标注信息
        bbox = parse_filename(filename)
        if bbox is None:
            print(f"无法解析文件名: {filename}")
            skipped_count += 1
            continue
            
        # 转换为YOLO格式
        yolo_annotation = convert_to_yolo_format(width, height, bbox)
        
        # 保存图像和标注文件
        base_name = os.path.splitext(filename)[0]
        new_img_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # 复制图像（你也可以选择移动）
        cv2.imwrite(new_img_path, img)
        
        # 写入标注文件
        with open(label_path, 'w') as f:
            f.write(yolo_annotation)
            
        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"已处理 {processed_count} 张图片...")
    
    print(f"处理完成! 成功: {processed_count}, 跳过: {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将CCPD数据集转换为YOLO格式')
    parser.add_argument('--input', type=str, required=True, help='输入目录（包含CCPD图片）')
    parser.add_argument('--output', type=str, required=True, help='输出目录（用于存放YOLO格式数据集）')
    
    args = parser.parse_args()
    process_dataset(args.input, args.output)