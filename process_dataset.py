"""
印章一致性验证 - 数据准备脚本
功能：从单一真章数据集构建配对训练数据，标签为1的配对表示的是同一个印章，而标签为0表示的是不同印章。具体来说，标签为0的负样本（不同印章的配对）是通过随机选择两张不同的印章图像来生成的。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
import random

class SealDataGenerator:
    """印章数据增强生成器 - 模拟扫描/拍摄变化"""
    
    def __init__(self):
        # 模拟扫描件的变换（轻度）
        self.scan_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), 
                     border_mode=cv2.BORDER_CONSTANT, cval=255, p=0.5),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.4),
        ])
        
        # 模拟预留印鉴（高清，变化更小）
        self.template_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        ])
        
    def generate_scan_pair(self, img):
        """生成扫描件-预留印鉴配对"""
        scan = self.scan_transform(image=img)['image']
        template = self.template_transform(image=img)['image']
        return scan, template
    
    def generate_different_pair(self, img1, img2):
        """生成不同印章的配对（负样本）"""
        scan = self.scan_transform(image=img1)['image']
        template = self.template_transform(image=img2)['image']
        return scan, template


def prepare_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    准备训练数据集
    
    Args:
        source_dir: 原始真章目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        for img_type in ['scan', 'template']:
            (output_path / split / img_type).mkdir(parents=True, exist_ok=True)
    
    # 读取所有图片
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    print(f"找到 {len(image_files)} 张印章图片")   # 原数据集1223张
    
    # 划分数据集
    random.shuffle(image_files)
    n_train = int(len(image_files) * train_ratio)  # 856
    n_val = int(len(image_files) * val_ratio)  # 183
    
    splits = {
        'train': image_files[:n_train],  # 856
        'val': image_files[n_train:n_train+n_val], # 183
        'test': image_files[n_train+n_val:]  # 184
    }
    
    generator = SealDataGenerator()
    annotations = {'train': [], 'val': [], 'test': []}
    
    for split_name, files in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        # 为每个split生成正负样本
        for idx, img_file in enumerate(tqdm(files)):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # 生成正样本（同一印章的配对）
            scan, template = generator.generate_scan_pair(img)  # 每张图片生成一个扫描件。
            
            scan_name = f"{split_name}_{idx:06d}_scan.jpg"
            template_name = f"{split_name}_{idx:06d}_template.jpg"
            
            cv2.imwrite(str(output_path / split_name / 'scan' / scan_name), scan)
            cv2.imwrite(str(output_path / split_name / 'template' / template_name), template)
            
            annotations[split_name].append({  # 每张图片生成的扫描件和原图组成一对正样本。
                'scan': scan_name,
                'template': template_name,
                'label': 1,  # 正样本
                'pair_id': idx
            })
            
        # 生成负样本（不同印章的配对）
        print(f"生成 {split_name} 负样本...")
        n_negative = len(files)  # 负样本数量与正样本相同
        
        for idx in tqdm(range(n_negative)):
            # 随机选择两张不同的图片，组成负样本。
            idx1, idx2 = random.sample(range(len(files)), 2)
            img1 = cv2.imread(str(files[idx1]))
            img2 = cv2.imread(str(files[idx2]))
            
            if img1 is None or img2 is None:
                continue
            
            scan, template = generator.generate_different_pair(img1, img2)
            
            scan_name = f"{split_name}_{len(files)+idx:06d}_scan.jpg"
            template_name = f"{split_name}_{len(files)+idx:06d}_template.jpg"
            
            cv2.imwrite(str(output_path / split_name / 'scan' / scan_name), scan)
            cv2.imwrite(str(output_path / split_name / 'template' / template_name), template)
            
            annotations[split_name].append({
                'scan': scan_name,
                'template': template_name,
                'label': 0,  # 负样本
                'pair_id': len(files) + idx
            })
    
    # 保存标注文件
    for split_name in ['train', 'val', 'test']:
        with open(output_path / f'{split_name}_annotations.json', 'w') as f:
            json.dump(annotations[split_name], f, indent=2)
        print(f"{split_name} 集: {len(annotations[split_name])} 对样本")
    
    # 保存数据集信息
    dataset_info = {
        'total_seals': len(image_files),
        'train_pairs': len(annotations['train']),
        'val_pairs': len(annotations['val']),
        'test_pairs': len(annotations['test']),
        'positive_ratio': 0.5,
        'description': '印章一致性验证数据集 - 每对包含扫描件和预留印鉴'
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n数据集准备完成!")
    print(f"输出目录: {output_dir}")
    print(f"总样本对数: {sum(len(v) for v in annotations.values())}")


if __name__ == "__main__":
    # 配置路径
    SOURCE_DIR = "dataset/盖章数据集/真章/居中"
    OUTPUT_DIR = "./seal_verification_dataset"
    
    # 准备数据集
    prepare_dataset(SOURCE_DIR, OUTPUT_DIR)
    
    print("\n下一步:")
    print("1. 运行此脚本生成配对数据")
    print("2. 使用训练脚本训练模型")
    print("3. 使用推理脚本进行验证")