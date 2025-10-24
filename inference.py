"""
印章一致性验证 - 推理和可视化脚本（增强版）
支持保存每对数据的详细结果（相似度、一致性概率、预测标签等）
"""

import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import seaborn as sns


class SealVerifier:
    """印章验证推理器（增强版）"""
    
    def __init__(self, model_path, config=None, device=None):
        """
        Args:
            model_path: 模型权重文件路径
            config: 模型配置（如果为None则从checkpoint加载）
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config is None:
            config = checkpoint.get('config', {})
        
        self.config = config
        
        # 创建模型
        from seal_model import create_model
        self.model = create_model(
            backbone=config.get('backbone', 'resnet50'),
            embedding_dim=config.get('embedding_dim', 512),
            pretrained=False
        )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功!")
        print(f"设备: {self.device}")
        print(f"Checkpoint来自 Epoch {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            print(f"验证指标: Acc={checkpoint['metrics'].get('accuracy', 0):.4f}, "
                  f"AUC={checkpoint['metrics'].get('auc', 0):.4f}")
    
    @torch.no_grad()
    def verify_pair(self, scan_path, template_path, visualize=True):
        """
        验证单对印章
        
        Args:
            scan_path: 扫描印章图路径
            template_path: 预留印鉴图路径
            visualize: 是否可视化结果
        
        Returns:
            dict: 包含相似度、预测结果等信息
        """
        from seal_dataset import InferenceDataset
        
        # 加载图像
        dataset = InferenceDataset(scan_path, template_path, 
                                   img_size=self.config.get('img_size', 256))
        scan_tensor, template_tensor = dataset.get_pair()
        
        # 移到设备
        scan_tensor = scan_tensor.to(self.device)
        template_tensor = template_tensor.to(self.device)
        
        # 推理计时
        start_time = time.time()
        similarity, emb1, emb2 = self.model(scan_tensor, template_tensor)
        inference_time = time.time() - start_time
        
        # 结果
        similarity_score = similarity.item()
        prediction = 1 if similarity_score > 0.5 else 0
        confidence = similarity_score if prediction == 1 else (1 - similarity_score)
        
        result = {
            'similarity': similarity_score,
            'prediction': prediction,
            'prediction_label': '一致' if prediction == 1 else '不一致',
            'confidence': confidence,
            'inference_time': inference_time,
            'scan_path': str(scan_path),
            'template_path': str(template_path)
        }
        
        # 可视化
        if visualize:
            self._visualize_result(scan_path, template_path, result)
        
        return result
    
    def _visualize_result(self, scan_path, template_path, result):
        """可视化验证结果"""
        # 读取原始图像
        scan_img = cv2.imread(str(scan_path))
        template_img = cv2.imread(str(template_path))
        
        scan_img = cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 扫描图
        axes[0].imshow(scan_img)
        axes[0].set_title('Scan Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 预留印鉴
        axes[1].imshow(template_img)
        axes[1].set_title('Template Image', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 结果显示
        axes[2].axis('off')
        result_text = (
            f"Similarity Score: {result['similarity']:.4f}\n\n"
            f"Prediction: {result['prediction_label']}\n\n"
            f"Confidence: {result['confidence']:.2%}\n\n"
            f"Inference Time: {result['inference_time']*1000:.2f} ms"
        )
        
        color = 'green' if result['prediction'] == 1 else 'red'
        axes[2].text(0.5, 0.5, result_text, 
                    ha='center', va='center',
                    fontsize=14,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        axes[2].set_title('Verification Result', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @torch.no_grad()
    def evaluate_testset(self, test_loader, save_dir='./eval_results', 
                        save_detailed_results=True):
        """
        评估测试集（增强版 - 保存详细结果）
        
        Args:
            test_loader: 测试数据加载器
            save_dir: 结果保存目录
            save_detailed_results: 是否保存每对数据的详细结果
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_probs = []
        all_preds = []
        all_labels = []
        inference_times = []
        
        # 用于保存详细结果
        detailed_results = []
        
        print("评估测试集...")
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            scan = batch['scan'].to(self.device)
            template = batch['template'].to(self.device)
            labels = batch['label']
            
            # 获取图像路径（如果有）
            scan_paths = batch.get('scan_path', [f'scan_{batch_idx}_{i}' for i in range(len(labels))])
            template_paths = batch.get('template_path', [f'template_{batch_idx}_{i}' for i in range(len(labels))])
            
            start_time = time.time()
            similarity, _, _ = self.model(scan, template)
            batch_time = time.time() - start_time
            inference_times.append(batch_time)
            
            probs = similarity.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # 保存每个样本的详细结果
            if save_detailed_results:
                for i in range(len(labels)):
                    prob = float(probs[i]) if probs.ndim > 0 else float(probs)
                    pred = int(preds[i]) if preds.ndim > 0 else int(preds)
                    label = int(labels[i])
                    
                    detailed_results.append({
                        'scan_path': scan_paths[i],
                        'template_path': template_paths[i],
                        'true_label': label,
                        'true_label_text': '一致' if label == 1 else '不一致',
                        'similarity_score': prob,
                        'consistency_probability': prob,  # 一致性概率 = 相似度分数
                        'predicted_label': pred,
                        'predicted_label_text': '一致' if pred == 1 else '不一致',
                        'confidence': prob if pred == 1 else (1 - prob),
                        'is_correct': (pred == label),
                        'inference_time_ms': (batch_time / len(labels)) * 1000
                    })
            
            all_probs.extend(probs if probs.ndim > 0 else [probs])
            all_preds.extend(preds if preds.ndim > 0 else [preds])
            all_labels.extend(labels.numpy())
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs),
            'avg_inference_time': np.mean(inference_times),
            'total_samples': len(all_labels)
        }
        
        print("\n测试集评估结果:")
        print(f"准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"平均推理时间: {metrics['avg_inference_time']*1000:.2f} ms/batch")
        print(f"总样本数: {metrics['total_samples']}")
        
        # 保存指标
        with open(save_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存详细结果
        if save_detailed_results and detailed_results:
            # 保存为CSV
            df = pd.DataFrame(detailed_results)
            csv_path = save_dir / 'detailed_results.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n详细结果已保存至CSV: {csv_path}")
            
            # 保存为JSON
            json_path = save_dir / 'detailed_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            print(f"详细结果已保存至JSON: {json_path}")
            
            # 打印统计信息
            print(f"\n详细结果统计:")
            print(f"总样本数: {len(detailed_results)}")
            print(f"正确预测: {sum(r['is_correct'] for r in detailed_results)}")
            print(f"错误预测: {sum(not r['is_correct'] for r in detailed_results)}")
            
            # 打印前5个样本示例
            print(f"\n前5个样本示例:")
            for i, result in enumerate(detailed_results[:5]):
                print(f"\n样本 {i+1}:")
                print(f"  扫描图: {result['scan_path']}")
                print(f"  模板图: {result['template_path']}")
                print(f"  真实标签: {result['true_label_text']}")
                print(f"  相似度分数: {result['similarity_score']:.4f}")
                print(f"  一致性概率: {result['consistency_probability']:.4f}")
                print(f"  预测标签: {result['predicted_label_text']}")
                print(f"  置信度: {result['confidence']:.4f}")
                print(f"  预测正确: {'✓' if result['is_correct'] else '✗'}")
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(all_labels, all_preds, save_dir)
        
        # 绘制ROC曲线
        self._plot_roc_curve(all_labels, all_probs, save_dir)
        
        # 绘制相似度分布
        self._plot_score_distribution(all_labels, all_probs, save_dir)
        
        return metrics, detailed_results
    
    def _plot_confusion_matrix(self, labels, preds, save_dir):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['不一致', '一致'],
                    yticklabels=['不一致', '一致'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"混淆矩阵已保存至: {save_dir / 'confusion_matrix.png'}")
    
    def _plot_roc_curve(self, labels, probs, save_dir):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curve.png', dpi=300)
        plt.close()
        
        print(f"ROC曲线已保存至: {save_dir / 'roc_curve.png'}")
    
    def _plot_score_distribution(self, labels, probs, save_dir):
        """绘制相似度分数分布"""
        labels = np.array(labels)
        probs = np.array(probs)
        
        plt.figure(figsize=(10, 6))
        
        # 正样本分布
        positive_scores = probs[labels == 1]
        plt.hist(positive_scores, bins=50, alpha=0.6, label='一致样本', color='green')
        
        # 负样本分布
        negative_scores = probs[labels == 0]
        plt.hist(negative_scores, bins=50, alpha=0.6, label='不一致样本', color='red')
        
        # 阈值线
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='阈值 (0.5)')
        
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'score_distribution.png', dpi=300)
        plt.close()
        
        print(f"分数分布图已保存至: {save_dir / 'score_distribution.png'}")


def demo_inference():
    """演示推理流程"""
    
    # 配置
    MODEL_PATH = './outputs/seal_verification/best_model.pth'
    SCAN_IMG = './examples/scan_001.jpg'
    TEMPLATE_IMG = './examples/template_001.jpg'
    
    # 创建验证器
    verifier = SealVerifier(MODEL_PATH)
    
    # 单对验证
    print("\n=== 单对印章验证 ===")
    result = verifier.verify_pair(SCAN_IMG, TEMPLATE_IMG, visualize=True)
    
    print(f"\n验证结果:")
    print(f"相似度: {result['similarity']:.4f}")
    print(f"预测: {result['prediction_label']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"推理时间: {result['inference_time']*1000:.2f} ms")


def evaluate_model():
    """评估模型性能（保存详细结果）"""
    
    from seal_dataset import create_dataloaders
    
    # 配置
    MODEL_PATH = './outputs/seal_verification3/best_model.pth'
    DATA_DIR = './seal_verification_dataset'
    
    # 创建验证器
    verifier = SealVerifier(MODEL_PATH)
    
    # 加载测试集
    _, _, test_loader = create_dataloaders(
        DATA_DIR,
        batch_size=32,
        img_size=256,
        num_workers=4
    )
    
    # 评估（保存详细结果）
    print("\n=== 测试集评估 ===")
    metrics, detailed_results = verifier.evaluate_testset(
        test_loader, 
        save_dir='./eval_results',
        save_detailed_results=True  # 启用详细结果保存
    )
    
    # 性能检查
    print("\n" + "="*60)
    if metrics['accuracy'] >= 0.95:
        print(f"✓ 模型达到目标准确率 (≥95%): {metrics['accuracy']:.4f}")
    else:
        print(f"✗ 模型未达到目标准确率 (≥95%): {metrics['accuracy']:.4f}")
    
    if metrics['avg_inference_time'] <= 5.0:
        print(f"✓ 推理时间符合要求 (≤5秒): {metrics['avg_inference_time']:.2f}秒")
    else:
        print(f"✗ 推理时间超出要求 (≤5秒): {metrics['avg_inference_time']:.2f}秒")
    print("="*60)
    
    return metrics, detailed_results


if __name__ == "__main__":
    import sys
    evaluate_model()   # 批量测试集测试
    # demo_inference()   # 测试一对