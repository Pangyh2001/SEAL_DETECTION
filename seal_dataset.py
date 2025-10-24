"""
印章一致性验证 - 数据集加载器（修改版 - 返回完整路径）
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SealVerificationDataset(Dataset):
    """印章验证数据集"""
    
    def __init__(self, data_dir, split='train', transform=None, img_size=256):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            transform: 数据增强
            img_size: 图像大小
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # 加载标注文件
        annotation_file = self.data_dir / f'{split}_annotations.json'
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 设置数据增强
        if transform is None:
            if split == 'train':
                self.transform = self._get_train_transform()
            else:
                self.transform = self._get_val_transform()
        else:
            self.transform = transform
    
    def _get_train_transform(self):
        """训练集数据增强"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_val_transform(self):
        """验证/测试集数据增强"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _load_image(self, img_path):
        """加载并预处理图像"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """获取一对印章图像（修改版 - 返回完整路径）"""
        item = self.annotations[idx]
        
        # 构建图像路径
        scan_path = self.data_dir / self.split / 'scan' / item['scan']
        template_path = self.data_dir / self.split / 'template' / item['template']
        
        # 加载图像
        scan_img = self._load_image(scan_path)
        template_img = self._load_image(template_path)
        
        # 数据增强
        scan_img = self.transform(image=scan_img)['image']
        template_img = self.transform(image=template_img)['image']
        
        # 标签
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # ✅ 修改：返回完整路径而不是文件名
        return {
            'scan': scan_img,
            'template': template_img,
            'label': label,
            'scan_path': str(scan_path),        # ✅ 返回完整路径
            'template_path': str(template_path), # ✅ 返回完整路径
            'scan_name': item['scan'],           # 保留文件名（可选）
            'template_name': item['template']    # 保留文件名（可选）
        }


def collate_fn_with_paths(batch):
    """
    ✅ 自定义collate函数，保留路径信息
    """
    scans = torch.stack([item['scan'] for item in batch])
    templates = torch.stack([item['template'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # 保留路径列表
    scan_paths = [item['scan_path'] for item in batch]
    template_paths = [item['template_path'] for item in batch]
    
    # 可选：保留文件名
    scan_names = [item['scan_name'] for item in batch]
    template_names = [item['template_name'] for item in batch]
    
    return {
        'scan': scans,
        'template': templates,
        'label': labels,
        'scan_path': scan_paths,           # ✅ 完整路径
        'template_path': template_paths,   # ✅ 完整路径
        'scan_name': scan_names,           # 文件名
        'template_name': template_names    # 文件名
    }


def create_dataloaders(data_dir, batch_size=16, img_size=256, num_workers=4):
    """创建数据加载器（修改版 - 使用自定义collate函数）"""
    
    # 创建数据集
    train_dataset = SealVerificationDataset(data_dir, split='train', img_size=img_size)
    val_dataset = SealVerificationDataset(data_dir, split='val', img_size=img_size)
    test_dataset = SealVerificationDataset(data_dir, split='test', img_size=img_size)
    
    # ✅ 创建数据加载器 - 添加 collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_with_paths  # ✅ 使用自定义collate函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_paths  # ✅ 使用自定义collate函数
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_paths  # ✅ 使用自定义collate函数
    )
    
    return train_loader, val_loader, test_loader


class InferenceDataset(Dataset):
    """推理用数据集 - 单对图像"""
    
    def __init__(self, scan_path, template_path, img_size=256):
        self.scan_path = scan_path
        self.template_path = template_path
        self.img_size = img_size
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _load_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_pair(self):
        """获取图像对"""
        scan_img = self._load_image(self.scan_path)
        template_img = self._load_image(self.template_path)
        
        scan_img = self.transform(image=scan_img)['image']
        template_img = self.transform(image=template_img)['image']
        
        # 添加batch维度
        scan_img = scan_img.unsqueeze(0)
        template_img = template_img.unsqueeze(0)
        
        return scan_img, template_img


if __name__ == "__main__":
    # 测试数据集
    data_dir = "./seal_verification_dataset"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, 
        batch_size=8,
        img_size=256,
        num_workers=2
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # ✅ 测试数据加载（包含路径）
    batch = next(iter(test_loader))
    print(f"\n批次数据:")
    print(f"扫描图形状: {batch['scan'].shape}")
    print(f"模板图形状: {batch['template'].shape}")
    print(f"标签形状: {batch['label'].shape}")
    print(f"标签分布: {batch['label'].sum().item()}/{len(batch['label'])}")
    
    # ✅ 检查路径信息
    print(f"\n路径信息检查:")
    if 'scan_path' in batch:
        print(f"✅ 包含完整路径信息")
        print(f"示例扫描图路径: {batch['scan_path'][0]}")
        print(f"示例模板图路径: {batch['template_path'][0]}")
        print(f"示例文件名: {batch['scan_name'][0]}")
    else:
        print(f"❌ 未包含路径信息")