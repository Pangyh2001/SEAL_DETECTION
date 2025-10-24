"""
印章一致性验证 - 训练脚本
支持断点续训、学习率调度、早停等功能
"""

import os
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块（需要在同目录下）
# from seal_model import create_model, ContrastiveLoss
# from seal_dataset import create_dataloaders


class Trainer:
    """印章验证模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        from seal_model import ContrastiveLoss
        self.criterion = ContrastiveLoss(
            margin=config.get('margin', 1.0),
            bce_weight=config.get('bce_weight', 0.5)
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        if config.get('scheduler', 'cosine') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        # 早停
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            scan = batch['scan'].to(self.device)
            template = batch['template'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            similarity, emb1, emb2 = self.model(scan, template)
            
            # 计算损失
            loss, bce_loss, contrastive_loss = self.criterion(
                similarity, emb1, emb2, labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('clip_grad', 0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
            
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            preds = (similarity.squeeze() > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bce': f"{bce_loss.item():.4f}",
                'contrast': f"{contrastive_loss.item():.4f}"
            })
        
        # 计算指标
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
        
        for batch in pbar:
            scan = batch['scan'].to(self.device)
            template = batch['template'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            similarity, emb1, emb2 = self.model(scan, template)
            
            # 计算损失
            loss, _, _ = self.criterion(similarity, emb1, emb2, labels)
            
            running_loss += loss.item()
            
            # 统计
            probs = similarity.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 计算指标
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1,
            'auc': epoch_auc
        }
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练...")
        print(f"设备: {self.device}")
        print(f"训练批次: {len(self.train_loader)}")
        print(f"验证批次: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['accuracy'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rates'].append(current_lr)
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"✓ 保存最佳模型 (Acc: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config.get('patience', 15):
                print(f"\n早停触发! 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})")
                break
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)
            
            print("-" * 80)
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})")
        
        # 保存训练历史
        self.save_history()
        self.plot_history()
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
    
    def save_history(self):
        """保存训练历史"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练历史已保存至: {history_file}")
    
    def plot_history(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC曲线
        axes[1, 0].plot(self.history['val_auc'], label='Val AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(self.history['learning_rates'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"训练曲线已保存至: {save_path}")


def main():
    """主训练流程"""
    
    # 训练配置
    config = {
        # 数据配置
        'data_dir': './seal_verification_dataset',
        'batch_size': 16,
        'img_size': 256,
        'num_workers': 4,
        
        # 模型配置
        'backbone': 'resnet50',  # 'resnet50' 或 'efficientnet_b3'
        'embedding_dim': 512,
        'pretrained': True,
        
        # 训练配置
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'clip_grad': 1.0,
        
        # 损失配置
        'margin': 1.0,
        'bce_weight': 0.6,
        
        # 学习率调度
        'scheduler': 'cosine',  # 'cosine' 或 'plateau'
        'min_lr': 1e-6,
        
        # 早停和保存
        'patience': 15,
        'save_interval': 10,
        
        # 输出目录
        'output_dir': './outputs/seal_verification2'
    }
    
    # 导入模块
    from seal_model import create_model
    from seal_dataset import create_dataloaders
    
    # 创建数据加载器
    print("加载数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("创建模型...")
    model = create_model(
        backbone=config['backbone'],
        embedding_dim=config['embedding_dim'],
        pretrained=config['pretrained']
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 开始训练
    history = trainer.train()
    
    print("\n训练完成!")
    print(f"模型和日志保存在: {config['output_dir']}")


if __name__ == "__main__":
    main()