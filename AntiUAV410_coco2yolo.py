import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def process_coco_to_yolo_complete(coco_annotations_path, original_images_root, output_root):
    """
    完整的COCO转YOLO处理脚本,递归处理所有子文件夹并智能重命名图片
    """
    
    # 读取COCO标注文件
    with open(coco_annotations_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建图像ID到图像信息的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 创建类别映射
    categories = coco_data['categories']
    category_id_map = {}
    for i, cat in enumerate(sorted(categories, key=lambda x: x['id'])):
        category_id_map[cat['id']] = i
    
    # 按图像ID分组标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 递归查找所有图片文件
    print("扫描图片文件...")
    all_image_files = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    for root, dirs, files in os.walk(original_images_root):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                # 使用相对路径作为键，避免重复
                rel_path = os.path.relpath(full_path, original_images_root)
                all_image_files[rel_path] = full_path
    
    print(f"找到 {len(all_image_files)} 张图片文件")
    
    # 处理每个有标注的图像
    processed_count = 0
    missing_count = 0
    
    for image_id, image_info in tqdm(image_id_to_info.items(), desc="处理图片"):
        original_file_path = image_info['file_name']  # 如 'subfolder/image.jpg'
        image_width = image_info['width']
        image_height = image_info['height']
        
        # 查找图片文件
        source_image_path = None
        for rel_path, full_path in all_image_files.items():
            if rel_path.replace('\\', '/') == original_file_path:
                source_image_path = full_path
                break
        
        if not source_image_path or not os.path.exists(source_image_path):
            missing_count += 1
            continue
        
        # 提取子文件夹名称用于重命名
        folder_name = os.path.dirname(original_file_path)
        if not folder_name:  # 如果在根目录
            folder_name = "root"
        else:
            # 取最后一级目录名，避免路径过长
            folder_name = os.path.basename(folder_name)
        
        # 生成新的文件名：子文件夹名_原文件名
        original_filename = os.path.basename(original_file_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        new_filename = f"{folder_name}_{name_without_ext}"
        
        # 目标路径
        image_output_dir = os.path.join(output_root, 'images')
        label_output_dir = os.path.join(output_root, 'labels')
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)
        
        # 新文件路径
        new_image_path = os.path.join(image_output_dir, f"{new_filename}.jpg")
        new_label_path = os.path.join(label_output_dir, f"{new_filename}.txt")
        
        try:
            # 复制图片文件
            shutil.copy2(source_image_path, new_image_path)
            
            # 生成YOLO格式标签
            with open(new_label_path, 'w', encoding='utf-8') as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        coco_category_id = ann['category_id']
                        if coco_category_id not in category_id_map:
                            continue
                            
                        yolo_class_id = category_id_map[coco_category_id]
                        bbox = ann['bbox']
                        
                        # COCO转YOLO格式转换
                        x_center = (bbox[0] + bbox[2] / 2) / image_width
                        y_center = (bbox[1] + bbox[3] / 2) / image_height
                        width_norm = bbox[2] / image_width
                        height_norm = bbox[3] / image_height
                        
                        # 确保值在有效范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width_norm = max(0, min(1, width_norm))
                        height_norm = max(0, min(1, height_norm))
                        
                        f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理图片 {original_file_path} 时出错: {str(e)}")
            # 清理可能已创建的文件
            for path in [new_image_path, new_label_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 张图片")
    print(f"缺失图片: {missing_count} 张")
    print(f"输出目录: {output_root}")

def find_all_annotations_files(dataset_root):
    """
    在数据集根目录中查找所有可能的标注文件
    """
    annotations_dir = os.path.join(dataset_root, 'annotations')
    if not os.path.exists(annotations_dir):
        return []
    
    annotation_files = []
    for file in os.listdir(annotations_dir):
        if file.endswith('.json'):
            annotation_files.append(os.path.join(annotations_dir, file))
    
    return annotation_files

def main():
    """
    主函数 - 配置参数并执行转换
    """
    # 配置参数
    DATASET_ROOT = "/path/to/your/dataset"  # 修改为您的数据集根目录
    OUTPUT_ROOT = "/path/to/output/yolo_dataset"  # 修改为输出目录
    
    # 在annotations文件夹中查找标注文件
    annotation_files = find_all_annotations_files(DATASET_ROOT)
    
    if not annotation_files:
        print("在annotations目录中未找到任何.json标注文件")
        return
    
    print("找到以下标注文件:")
    for i, f in enumerate(annotation_files):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    # 处理每个标注文件
    for annotation_file in annotation_files:
        print(f"\n处理标注文件: {os.path.basename(annotation_file)}")
        
        # 根据标注文件名确定数据集划分
        filename = os.path.basename(annotation_file).lower()
        if 'train' in filename:
            split_name = 'train'
            images_subdir = 'train'
        elif 'val' in filename or 'valid' in filename:
            split_name = 'val'
            images_subdir = 'val'
        elif 'test' in filename:
            split_name = 'test'
            images_subdir = 'test'
        else:
            split_name = 'unknown'
            images_subdir = 'images'  # 默认
        
        # 原始图片目录
        original_images_dir = os.path.join(DATASET_ROOT, images_subdir)
        
        if not os.path.exists(original_images_dir):
            print(f"警告: 图片目录不存在: {original_images_dir}")
            continue
        
        # 输出目录
        split_output_dir = os.path.join(OUTPUT_ROOT, split_name)
        
        # 执行转换
        process_coco_to_yolo_complete(
            annotation_file,
            original_images_dir,
            split_output_dir
        )

if __name__ == "__main__":
    main()