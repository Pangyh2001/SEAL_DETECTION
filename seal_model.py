"""
印章一致性验证 - Siamese Network模型
支持多尺度特征提取和相似度计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            # 提取多层特征
            self.layer1 = nn.Sequential(*list(base_model.children())[:5])  # 256
            self.layer2 = base_model.layer2  # 512
            self.layer3 = base_model.layer3  # 1024
            self.layer4 = base_model.layer4  # 2048
            
            feature_dims = [256, 512, 1024, 2048]
            
        elif backbone == 'efficientnet_b3':
            base_model = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
            self.backbone = base_model
            feature_dims = base_model.feature_info.channels()  # [24, 32, 48, 136, 384]
            
        self.backbone_type = backbone
        self.feature_dims = feature_dims
        
        # 自适应池化到固定大小
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        
    def forward(self, x):
        """提取多尺度特征"""
        features = []
        
        if self.backbone_type == 'resnet50':
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            multi_scale = [x1, x2, x3, x4]
        else:
            multi_scale = self.backbone(x)
            multi_scale = multi_scale[-4:]  # 取最后4个尺度
        
        # 对每个尺度应用自适应池化
        for feat, pool in zip(multi_scale, self.adaptive_pools):
            pooled = pool(feat)
            features.append(pooled.flatten(1))
        
        return features


class AttentionModule(nn.Module):
    """注意力模块 - 用于特征加权"""
    
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class SiameseSealNetwork(nn.Module):
    """
    孪生网络用于印章一致性验证
    支持多尺度特征提取和相似度计算
    """
    
    def __init__(self, backbone='resnet50', embedding_dim=512, pretrained=True):
        super().__init__()
        
        # 多尺度特征提取器（共享权重）
        self.feature_extractor = MultiScaleFeatureExtractor(backbone, pretrained)
        
        # 计算总特征维度
        total_dim = sum([dim * (pool.output_size[0] ** 2) 
                        for dim, pool in zip(self.feature_extractor.feature_dims, 
                                            self.feature_extractor.adaptive_pools)])
        
        # 注意力模块
        self.attention = AttentionModule(total_dim)
        
        # 特征投影到嵌入空间
        self.projection = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # 相似度计算头
        self.similarity_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),  # concat + diff + element-wise product
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward_once(self, x):
        """单次前向传播 - 提取特征"""
        # 多尺度特征提取
        multi_scale_features = self.feature_extractor(x)
        
        # 拼接所有尺度特征
        combined_features = torch.cat(multi_scale_features, dim=1)
        
        # 注意力加权
        attended_features = self.attention(combined_features)
        
        # 投影到嵌入空间
        embedding = self.projection(attended_features)
        
        return embedding
    
    def forward(self, img1, img2):
        """
        前向传播
        
        Args:
            img1: 扫描印章图 [B, 3, H, W]
            img2: 预留印鉴图 [B, 3, H, W]
        
        Returns:
            similarity: 相似度概率 [B, 1]
            embedding1: 图1的嵌入向量
            embedding2: 图2的嵌入向量
        """
        # 提取特征嵌入
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        
        # 计算多种特征交互
        concat_features = torch.cat([embedding1, embedding2], dim=1)
        diff_features = torch.abs(embedding1 - embedding2)
        prod_features = embedding1 * embedding2
        
        combined = torch.cat([concat_features, diff_features, prod_features], dim=1)
        
        # 相似度预测
        similarity = self.similarity_head(combined)
        
        return similarity, embedding1, embedding2
    
    def predict_similarity(self, img1, img2):
        """预测相似度（推理模式）"""
        self.eval()
        with torch.no_grad():
            similarity, _, _ = self.forward(img1, img2)
        return similarity


class ContrastiveLoss(nn.Module):
    """对比损失 + BCE损失的组合"""
    
    def __init__(self, margin=1.0, bce_weight=0.5):
        super().__init__()
        self.margin = margin
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
    
    def forward(self, similarity, embedding1, embedding2, labels):
        """
        Args:
            similarity: 预测的相似度 [B, 1]
            embedding1, embedding2: 特征嵌入 [B, D]
            labels: 真实标签 [B] (1为相似，0为不相似)
        """
        # BCE损失
        bce = self.bce_loss(similarity.squeeze(), labels.float())
        
        # 对比损失
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        contrastive = torch.mean(
            labels.float() * torch.pow(euclidean_distance, 2) +
            (1 - labels.float()) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        # 组合损失
        total_loss = self.bce_weight * bce + (1 - self.bce_weight) * contrastive
        
        return total_loss, bce, contrastive


def create_model(backbone='resnet50', embedding_dim=512, pretrained=True):
    """创建模型的工厂函数"""
    model = SiameseSealNetwork(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_model(backbone='resnet50', embedding_dim=512)
    
    # 模拟输入
    img1 = torch.randn(2, 3, 256, 256)
    img2 = torch.randn(2, 3, 256, 256)
    
    # 前向传播
    similarity, emb1, emb2 = model(img1, img2)
    
    print(f"模型结构测试:")
    print(f"输入形状: {img1.shape}")
    print(f"相似度输出: {similarity.shape}")
    print(f"嵌入向量: {emb1.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")