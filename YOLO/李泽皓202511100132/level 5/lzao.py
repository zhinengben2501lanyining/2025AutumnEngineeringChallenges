from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from difflib import SequenceMatcher

# 初始化模型
yolo_model = YOLO(r'C:\Users\99597\xiangmuone\CCPD\runs\detect\train8\weights\best.pt')  
ocr = PaddleOCR(use_textline_orientation=True, lang='ch')  

# 中国省份简称列表
PROVINCE_ABBREVIATIONS = [
    "京", "津", "沪", "渝", "冀", "晋", "辽", "吉", "黑", "苏", 
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "琼", 
    "川", "贵", "云", "陕", "甘", "青", "蒙", "桂", "宁", "新", 
    "藏", "使", "领", "警", "学", "港", "澳"
]

def find_best_province_match(first_char):
    """从省份简称列表中找出与给定字符最匹配的省份简称"""
    best_match = ""
    best_score = 0
    
    for province in PROVINCE_ABBREVIATIONS:
        # 计算相似度得分
        similarity = SequenceMatcher(None, first_char, province).ratio()
        
        # 如果相似度高于当前最佳匹配，则更新最佳匹配
        if similarity > best_score:
            best_score = similarity
            best_match = province
    
    # 设置一个阈值，只有相似度超过阈值才认为是有效匹配
    threshold = 0.3  # 可以根据实际情况调整这个阈值
    if best_score >= threshold:
        return best_match
    else:
        return None

def recognize_province_char(plate_img):
    """专门识别车牌第一个字符（省份简称）"""
    # 裁剪车牌左侧1/3区域（省份简称通常在左侧）
    height, width = plate_img.shape[:2]
    left_region = plate_img[:, :width//3]
    
    # 使用OCR识别左侧区域
    result = ocr.predict(left_region)
    
    # 解析结果
    if result and len(result) > 0:
        ocr_dict = result[0]
        if 'rec_texts' in ocr_dict and len(ocr_dict['rec_texts']) > 0:
            first_char = ocr_dict['rec_texts'][0]
            if first_char:  # 确保不是空字符串
                # 取第一个字符
                if len(first_char) > 0:
                    return first_char[0]
    
    # 如果没有识别到字符，返回None
    return None

def recognize_plate(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径:", image_path)
        return
    
    # 复制原始图像用于显示
    display_image = image.copy()
    
    # YOLO检测车牌
    results = yolo_model(image)
    
    # 用于存储所有识别结果
    all_plates = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 裁剪车牌区域
            plate_img = image[y1:y2, x1:x2]
            
            # 1. 识别车牌第一个字符（省份简称）
            first_char = recognize_province_char(plate_img)
            print(f"识别到的第一个字符: '{first_char}'")
            
            # 2. 使用OCR识别整个车牌
            result = ocr.predict(plate_img)
            
            # 处理OCR结果
            plate_text = ""
            confidence = 0.0
            
            if result and len(result) > 0:
                ocr_dict = result[0]
                if 'rec_texts' in ocr_dict and len(ocr_dict['rec_texts']) > 0:
                    plate_text = ocr_dict['rec_texts'][0]
                
                if 'rec_scores' in ocr_dict and len(ocr_dict['rec_scores']) > 0:
                    confidence = ocr_dict['rec_scores'][0]
            
            # 3. 处理省份简称
            province_prefix = ""
            if first_char:
                # 从省份简称列表中找出最匹配的省份
                best_province = find_best_province_match(first_char)
                if best_province:
                    province_prefix = best_province
                    print(f"匹配到的省份简称: '{best_province}'")
                else:
                    print(f"未找到匹配的省份简称")
            
            # 4. 组合最终的车牌号
            final_plate_text = plate_text
            if province_prefix:
                # 如果车牌文本中不包含省份简称，则添加
                if not plate_text.startswith(province_prefix):
                    final_plate_text = province_prefix + plate_text
            
            print(f"最终识别结果: '{final_plate_text}' (置信度: {confidence:.2f})")
            
            # 在显示图像上绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 如果识别到文本，则显示
            if final_plate_text:
                # 添加到结果列表
                all_plates.append((final_plate_text, confidence, (x1, y1, x2, y2)))
                
                # 计算文本大小
                font_scale = 1.5
                thickness = 4
                (text_width, text_height), baseline = cv2.getTextSize(final_plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # 确保文本不会超出图像边界
                text_x = x1
                text_y = y1 - 20
                
                if text_y < text_height + 20:
                    text_y = y2 + text_height + 30
                
                if text_x + text_width > display_image.shape[1]:
                    text_x = display_image.shape[1] - text_width - 10
                
                # 绘制文本背景
                overlay = display_image.copy()
                cv2.rectangle(overlay, (text_x - 10, text_y - text_height - 15), 
                             (text_x + text_width + 10, text_y + 15), (0, 0, 0), -1)
                
                # 将半透明背景添加到图像上
                alpha = 0.7
                display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)
                
                # 绘制文本
                cv2.putText(display_image, final_plate_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    # 在图像顶部添加所有识别结果的汇总信息
    if all_plates:
        # 准备汇总文本
        summary_text = "识别结果: " + ", ".join([f"{text}({conf:.2f})" for text, conf, _ in all_plates])
        
        # 计算汇总文本大小
        (summary_width, summary_height), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        
        # 确保汇总文本不会超出图像边界
        summary_x = (display_image.shape[1] - summary_width) // 2
        summary_y = 40
        
        # 绘制汇总文本背景
        cv2.rectangle(display_image, (summary_x - 10, summary_y - summary_height - 10), 
                     (summary_x + summary_width + 10, summary_y + 10), (0, 0, 0), -1)
        
        # 绘制汇总文本
        cv2.putText(display_image, summary_text, (summary_x, summary_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # 创建结果图像窗口
    cv2.namedWindow('License Plate Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('License Plate Recognition', 1000, 800)
    
    # 显示结果
    cv2.imshow('License Plate Recognition', display_image)
    
    # 等待用户按键
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果图像
    filename, ext = os.path.splitext(image_path)
    output_path = f"{filename}_result{ext}"
    cv2.imwrite(output_path, display_image)
    print(f"结果已保存到: {output_path}")
    
    # 在控制台再次显示识别结果
    print("\n=== 识别结果汇总 ===")
    print(f"原始图片: {image_path}")
    print(f"处理结果: {output_path}")
    if all_plates:
        for i, (text, conf, coords) in enumerate(all_plates):
            print(f"车牌 {i+1}: {text} (置信度: {conf:.2f}) 位置: {coords}")
    else:
        print("未识别到车牌")
    print("=====================")

# 使用示例
image_path = r"C:\Users\99597\xiangmuone\CCPD\mine\crv.jpg"  
recognize_plate(image_path)