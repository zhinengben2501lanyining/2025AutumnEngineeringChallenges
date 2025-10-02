import os
import cv2
import shutil
from tqdm import tqdm  # 进度条库（需安装：pip install tqdm）

def parse_ccpd_filename(filename):
    """
    解析 CCPD 图片文件名，提取 bounding box 坐标（x1,y1,x2,y2）
    :param filename: 不带后缀的图片文件名（如 "025-95_113-154&383_386&473-..."）
    :return: (x1, y1, x2, y2) 或 None（解析失败）
    """
    try:
        # 按 "-" 分割文件名，取第 3 个字段（x1&y1_x2&y2）
        parts = filename.split("-")
        bbox_str = parts[2]  # 示例："154&383_386&473"
        # 分割左上角和右下角坐标
        xy1, xy2 = bbox_str.split("_")
        x1, y1 = map(int, xy1.split("&"))
        x2, y2 = map(int, xy2.split("&"))
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"解析文件名失败：{filename}，错误：{e}")
        return None

def ccpd_to_yolo(ccpd_root, yolo_root, class_id=0):
    """
    将 CCPD 数据集转换为 YOLO 格式
    :param ccpd_root: CCPD 数据集根目录（如 "D:/datasets/CCPD2019"）
    :param yolo_root: 输出 YOLO 数据集根目录（如 "D:/datasets/CCPD_YOLO"）
    :param class_id: 车牌类别编号（默认 0，单类检测）
    """
    # 1. 创建 YOLO 数据集目录结构
    yolo_images = os.path.join(yolo_root, "images")  # 存放图片
    yolo_labels = os.path.join(yolo_root, "labels")  # 存放标注文件
    os.makedirs(yolo_images, exist_ok=True)
    os.makedirs(yolo_labels, exist_ok=True)

    # 2. 遍历 CCPD 所有子文件夹中的图片
    ccpd_subfolders = [f.path for f in os.scandir(ccpd_root) if f.is_dir()]
    total_files = 0
    # 先统计总文件数（用于进度条）
    for subfolder in ccpd_subfolders:
        total_files += len([f for f in os.listdir(subfolder) if f.endswith((".jpg", ".png"))])
    
    # 3. 逐个处理图片并生成标注
    with tqdm(total=total_files, desc="转换 CCPD 到 YOLO 格式") as pbar:
        for subfolder in ccpd_subfolders:
            for filename in os.listdir(subfolder):
                if not filename.endswith((".jpg", ".png")):
                    continue  # 跳过非图片文件
                
                # 3.1 读取图片，获取尺寸（用于归一化）
                img_path = os.path.join(subfolder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"跳过损坏图片：{img_path}")
                    pbar.update(1)
                    continue
                img_h, img_w = img.shape[:2]  # 图片高度、宽度
                
                # 3.2 解析文件名，获取 bounding box 坐标
                img_name_no_ext = os.path.splitext(filename)[0]
                bbox = parse_ccpd_filename(img_name_no_ext)
                if bbox is None:
                    pbar.update(1)
                    continue
                x1, y1, x2, y2 = bbox
                
                # 3.3 检查坐标有效性（避免超出图片范围）
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y2 = max(y1 + 1, min(y2, img_h))
                
                # 3.4 转换为 YOLO 格式（归一化）
                x_center = (x1 + x2) / (2 * img_w)  # 中心点 x 归一化
                y_center = (y1 + y2) / (2 * img_h)  # 中心点 y 归一化
                width = (x2 - x1) / img_w           # 宽度归一化
                height = (y2 - y1) / img_h          # 高度归一化
                
                # 3.5 保存图片到 YOLO images 目录
                dest_img_path = os.path.join(yolo_images, filename)
                shutil.copyfile(img_path, dest_img_path)  # 复制图片（避免移动原文件）
                
                # 3.6 保存标注到 YOLO labels 目录
                label_filename = img_name_no_ext + ".txt"
                label_path = os.path.join(yolo_labels, label_filename)
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                pbar.update(1)
    
    print(f"转换完成！YOLO 数据集保存至：{yolo_root}")
    print(f"图片数量：{len(os.listdir(yolo_images))}")
    print(f"标注文件数量：{len(os.listdir(yolo_labels))}")

if __name__ == "__main__":
    # -------------------------- 请修改以下路径 --------------------------
    CCPD_ROOT = "C:/Users/30735/Desktop/yolo/train"  # 你解压的 CCPD 根目录
    YOLO_ROOT = "C:/Users/30735/Desktop/datasets/CCPD"  # 输出 YOLO 数据集的根目录
    # -------------------------------------------------------------------
    ccpd_to_yolo(ccpd_root=CCPD_ROOT, yolo_root=YOLO_ROOT, class_id=0)