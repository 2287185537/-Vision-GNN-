"""Patch Selection and Processing Module

该模块实现了一个高效的patch选择和处理系统，包含以下核心组件：
1. PatchSelector: 基础patch选择器，使用轻量级多头注意力机制
2. ImportanceScorer: patch重要性评分器，结合位置编码
3. EnhancedPatchSelector: 增强型选择器，融合多种特征
4. ContentAwarePatchifier: 内容感知的patch处理器

技术特点：
- 使用混合精度训练提升性能
- 采用TorchScript加速关键计算
- 优化的内存使用和计算效率
"""

import configparser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 读取配置文件
conf = configparser.ConfigParser()
conf.read('/workspace/实验/output/confs/main.ini')

# 从配置文件获取模型的关键参数
PATCH_SIZE = int(conf['MODEL']['PATCH_SIZE'])  # patch的大小（边长）
DIMENSION = int(conf['MODEL']['DIMENSION'])     # 特征维度
HEAD_NUM = int(conf['MODEL']['HEAD_NUM'])      # 注意力头的数量

class PatchSelector(nn.Module):
    """轻量级多头注意力patch选择器
    
    技术特点：
    1. 采用线性投影+GELU激活的简化注意力机制
    2. 无偏置项设计减少参数量
    3. 高效的多头注意力融合策略
    
    计算流程：
    1. 输入特征经过LayerNorm归一化
    2. 并行计算多头注意力分数
    3. 使用1x1卷积高效融合多头信息
    4. 动态选择top-k个重要patches
    """
    def __init__(self, patch_dim=DIMENSION, num_heads=HEAD_NUM, top_k=None):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = patch_dim // num_heads
        
        # 轻量级注意力网络
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_dim, self.head_dim),
                nn.GELU(),
                nn.Linear(self.head_dim, 1, bias=False)  # 移除偏置项减少参数
            ) for _ in range(num_heads)
        ])
        
        # 简化的特征融合
        self.fusion = nn.Conv1d(num_heads, 1, 1, bias=False)
        self.layer_norm = nn.LayerNorm(patch_dim)
        self.top_k = top_k

    def forward(self, patches):
        """前向传播函数
        
        处理流程:
        1. 特征归一化
        2. 多头注意力计算
        3. 注意力分数融合
        4. 选择重要patches
        
        参数:
            patches (torch.Tensor): 输入特征图，形状为 [B, N, D]
                B: batch size
                N: patch数量
                D: 特征维度
                
        返回:
            selected_patches (torch.Tensor): 选中的patches，形状为 [B, K, D]
            attention_weights (torch.Tensor): 注意力权重，形状为 [B, N]
        """
        B, N, D = patches.shape
        
        # 确保top_k不超过patches数量
        if self.top_k is None:
            self.top_k = N // 2
        k = min(self.top_k, N)  # 防止k大于N
        
        # 合并归一化和注意力计算
        patches = self.layer_norm(patches)
        attention_weights = []
        
        # 批量处理所有头的计算
        for head in self.attention:
            scores = head(patches.reshape(-1, D)).reshape(B, N)
            attention_weights.append(scores)
        
        # 使用stack和conv1d进行高效融合
        attention_weights = torch.stack(attention_weights, dim=1)  # [B, H, N]
        fused_attention = self.fusion(attention_weights).squeeze(1)  # [B, N]
        attention_weights = F.softmax(fused_attention, dim=1)
        
        # 直接选择patches
        _, indices = torch.topk(attention_weights, k, dim=1)
        selected_patches = torch.gather(patches, 1,
            indices.unsqueeze(-1).expand(-1, -1, D))
            
        return selected_patches, attention_weights

class ImportanceScorer(nn.Module):
    """基于位置感知的patch重要性评分器
    
    技术亮点：
    1. 四分之一维度的轻量级评分网络
    2. 高效的正弦位置编码方案
    3. 无偏置设计降低计算量
    
    实现细节：
    - 使用正弦位置编码提供空间信息
    - 采用两层线性变换with GELU激活
    - Sigmoid输出确保分数在[0,1]范围
    """
    def __init__(self, patch_dim=DIMENSION):
        super().__init__()
        
        # 轻量级重要性评分网络
        self.importance_net = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 4),  # 大幅减少中间层维度
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 简化位置编码
        pe = self._create_position_embedding(196, patch_dim)
        self.register_buffer('pos_embedding', pe)
        
    def _create_position_embedding(self, length, dim):
        """生成正弦位置编码
        
        使用正弦和余弦函数生成位置编码，提供位置信息。
        
        参数:
            length (int): 序列长度，即patch数量
            dim (int): 编码维度，需与特征维度匹配
            
        返回:
            pe (torch.Tensor): 位置编码张量，形状为 [1, length, dim]
        """
        pos = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe
        
    def forward(self, patches):
        # 直接添加位置编码，无需dropout
        patches = patches + self.pos_embedding
        return self.importance_net(patches).squeeze(-1)

class EnhancedPatchSelector(PatchSelector):
    """高性能增强型patch选择器
    
    核心创新：
    1. 结合注意力机制和重要性评分的双路特征提取
    2. 使用混合精度计算加速训练
    3. 轻量级残差特征增强设计
    
    性能优化：
    - 并行计算减少延迟
    - 特征融合使用1x1卷积
    - 采用无偏置线性层减少参数
    """
    def __init__(self, patch_dim=DIMENSION, num_heads=HEAD_NUM, top_k=None):
        super().__init__(patch_dim, num_heads, top_k)
        
        self.importance_scorer = ImportanceScorer(patch_dim)
        
        # 轻量级特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Linear(patch_dim, patch_dim, bias=False),
            nn.GELU()
        )
        
        # 高效注意力融合
        self.attention_fusion = nn.Conv1d(2, 1, 1, bias=False)
        
    def forward(self, patches):
        """增强版本的前向传播
        
        特点:
        1. 并行计算提高效率
        2. 高效的特征融合
        3. 残差结构保持信息
        
        参数:
            patches (torch.Tensor): 输入特征，形状为 [B, N, D]
        
        返回:
            enhanced_patches (torch.Tensor): 增强后的特征，形状为 [B, K, D]
            final_scores (torch.Tensor): 最终的重要性分数，形状为 [B, N]
        """
        B, N, D = patches.shape
        
        # 确保维度匹配
        k = self.top_k if self.top_k is not None else (N // 2)
        k = min(k, N)  # 防止k大于N
        
        # 并行计算两种分数
        with torch.cuda.amp.autocast():  # 使用混合精度加速
            importance_scores = self.importance_scorer(patches)
            _, base_attention = super().forward(patches)
            
            # 快速融合
            scores = torch.stack([base_attention, importance_scores], dim=1)
            final_scores = self.attention_fusion(scores).squeeze(1)
            final_scores = F.softmax(final_scores, dim=1)
        
        # 高效选择
        _, indices = torch.topk(final_scores, k, dim=1)
        selected_patches = torch.gather(patches, 1,
            indices.unsqueeze(-1).expand(-1, -1, D))
        
        # 轻量级增强
        enhanced_patches = selected_patches + self.feature_enhancer(selected_patches)
        
        return enhanced_patches, final_scores

# 更新ContentAwarePatchifier使用增强版选择器
class ContentAwarePatchifier(nn.Module):
    """端到端的内容感知patch处理系统
    
    系统架构：
    1. 轻量级特征提取器
        - 四分之一维度的中间特征
        - 1x1卷积替代3x3降低计算量
        - 无偏置设计
        
    2. 高效patch处理流程
        - TorchScript加速张量操作
        - 混合精度训练
        - 优化的内存使用
        
    使用说明：
    - 输入要求：224x224分辨率图像
    - 输出：选定的重要patches特征
    - 可配置参数：patch大小、特征维度、头数
    
    优化设计：
    - 中间特征降维到DIMENSION//4
    - 使用1x1卷积替代3x3提升效率
    - 批处理优化减少内存占用
    """
    def __init__(self, in_channels=3, top_k="half"):
        super().__init__()
        
        # 高效特征提取
        mid_dim = DIMENSION // 4  # 减少中间特征维度
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim,
                     kernel_size=PATCH_SIZE,
                     stride=PATCH_SIZE,
                     bias=False),  # 移除偏置项
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, DIMENSION,
                     kernel_size=1,  # 使用1x1卷积代替3x3
                     bias=False)
        )
        
        # 计算每个patch的特征维度
        self.patch_dim = PATCH_SIZE * PATCH_SIZE * in_channels
        
        # 计算图像被划分后的patch数量
        self.num_patches = (224 // PATCH_SIZE) ** 2  # 输入图像大小为224x224
        
        # 调整特征维度
        self.feature_proj = nn.Linear(DIMENSION, self.patch_dim)
        
        # 设置top_k为patch总数的一半
        actual_top_k = self.num_patches // 2 if top_k == "half" else top_k
        
        self.selector = EnhancedPatchSelector(
            patch_dim=self.patch_dim,
            num_heads=HEAD_NUM,
            top_k=actual_top_k
        )
        
    def forward(self, x):
        # 获取batch size
        B = x.shape[0]
        
        # 使用torch.jit.script加速forward过程
        @torch.jit.script
        def process_patches(patches, B, H, W, C):
            return patches.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # patch划分和特征提取
        with torch.cuda.amp.autocast():
            patches = self.patchifier(x)
            B, C, H, W = patches.shape
            patches = process_patches(patches, B, H, W, C)
            patches = self.feature_proj(patches)
            selected_patches, _ = self.selector(patches)
        
        return selected_patches