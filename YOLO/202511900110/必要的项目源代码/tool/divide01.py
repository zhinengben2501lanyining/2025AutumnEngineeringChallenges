import os
import shutil
import random

def split_train_val(yolo_root, val_ratio=0.2):
    """
    拆分 YOLO 数据集为训练集和验证集
    :param yolo_root: YOLO 数据集根目录（如 "D:/datasets/CCPD_YOLO"）
    :param val_ratio: 验证集比例（默认 20%）
    """
    # 1. 创建训练/验证集目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_root, "labels", split), exist_ok=True)
    
    # 2. 获取所有图片文件名
    img_dir = os.path.join(yolo_root, "images")
    img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(img_files)  # 随机打乱
    
    # 3. 拆分并移动文件
    val_num = int(len(img_files) * val_ratio)
    val_files = img_files[:val_num]
    train_files = img_files[val_num:]
    
    for split, files in [("val", val_files), ("train", train_files)]:
        for filename in files:
            # 移动图片
            src_img = os.path.join(img_dir, filename)
            dst_img = os.path.join(img_dir, split, filename)
            shutil.move(src_img, dst_img)
            # 移动对应标注文件
            label_filename = os.path.splitext(filename)[0] + ".txt"
            src_label = os.path.join(yolo_root, "labels", label_filename)
            dst_label = os.path.join(yolo_root, "labels", split, label_filename)
            shutil.move(src_label, dst_label)
    
    # 4. 更新 data.yaml（拆分后路径）
    yaml_content = f"""train: ../CCPD_YOLO/images/train
val: ../CCPD_YOLO/images/val

nc: 1
names: ["license_plate"]
"""
    with open(os.path.join(yolo_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"拆分完成！训练集：{len(train_files)} 张，验证集：{len(val_files)} 张")

if __name__ == "__main__":
    split_train_val(yolo_root="D:/yolo model new 2/yolo/datasets/CCPD", val_ratio=0.2)