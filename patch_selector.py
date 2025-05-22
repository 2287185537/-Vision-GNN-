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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, patch_dim, num_heads, top_k=None):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = patch_dim // self.num_heads # Ensure this uses the parameter
        
        # 轻量级注意力网络
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_dim, self.head_dim),
                nn.GELU(),
                nn.Linear(self.head_dim, 1, bias=False)  # 移除偏置项减少参数
            ) for _ in range(self.num_heads) # Use self.num_heads
        ])
        
        # 简化的特征融合
        self.fusion = nn.Conv1d(self.num_heads, 1, 1, bias=False) # Use self.num_heads
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
    def __init__(self, patch_dim): # Removed default tied to global
        super().__init__()
        
        # 轻量级重要性评分网络
        self.importance_net = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 简化位置编码
        # Assuming 196 is a fixed length for now, e.g., (224/16)^2 = 14^2 = 196
        # If this needs to be dynamic, length should also be a parameter.
        pe = self._create_position_embedding(196, patch_dim) # Use patch_dim for dim
        self.register_buffer('pos_embedding', pe)
        
    def _create_position_embedding(self, length, dim): # dim comes from patch_dim
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
    def __init__(self, patch_dim, num_heads, top_k=None): # Removed defaults
        super().__init__(patch_dim, num_heads, top_k) # Pass params to parent
        
        self.importance_scorer = ImportanceScorer(patch_dim) # Pass patch_dim
        
        # 轻量级特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Linear(patch_dim, patch_dim, bias=False), # Use patch_dim
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
    - 中间特征降维到dimension//4
    - 使用1x1卷积替代3x3提升效率
    - 批处理优化减少内存占用
    """
    def __init__(self, patch_size, dimension, num_heads, in_channels=3, top_k="half"): # Added params
        super().__init__()
        
        # 高效特征提取
        mid_dim = dimension // 4  # Use parameter: dimension
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim,
                     kernel_size=patch_size, # Use parameter: patch_size
                     stride=patch_size,      # Use parameter: patch_size
                     bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, dimension,    # Use parameter: dimension
                     kernel_size=1,
                     bias=False)
        )
        
        # 计算图像被划分后的patch数量 (assuming image size 224x224 is fixed)
        self.num_patches = (224 // patch_size) ** 2  # Use parameter: patch_size
        
        # 调整特征维度 - This seems to project from `dimension` (from Conv2D) to `patch_dim_flattened`
        # If patchifier's output is already `dimension`, and selector expects `patch_dim_flattened`
        # that is `dimension`, then this projection might be redundant if DIMENSION was self.patch_dim_flattened.
        # The original code had: self.feature_proj = nn.Linear(DIMENSION, self.patch_dim)
        # where self.patch_dim = PATCH_SIZE * PATCH_SIZE * in_channels.
        # And EnhancedPatchSelector expected patch_dim = self.patch_dim.
        # The output of self.patchifier is C=DIMENSION.
        # So, the selector should expect `dimension` as its `patch_dim`.
        # Let's clarify:
        # The PatchSelector and its derivatives expect `patch_dim` as the feature dimension of each patch.
        # The `self.patchifier` outputs features of channel `dimension`.
        # The reshape `patches.permute(0, 2, 3, 1).reshape(B, H*W, C)` makes C (which is `dimension`) the feature dim.
        # So, `EnhancedPatchSelector` should be initialized with `patch_dim=dimension`.
        # The `self.feature_proj` seems to be an error in my previous reasoning or the original code's intent.
        # If `patchifier` outputs `dimension`, and selector takes `dimension`, then `feature_proj` is not needed
        # if its job was to map to what the selector expects.
        # Original: self.patch_dim = PATCH_SIZE * PATCH_SIZE * in_channels
        # Original: self.feature_proj = nn.Linear(DIMENSION, self.patch_dim)
        # Original selector took patch_dim = self.patch_dim
        # This implies the output of patchifier (DIMENSION) was projected to (PATCH_SIZE * PATCH_SIZE * in_channels)
        # This is unusual. Typically ConvNet features are used directly.
        # Let's stick to the original structure for now, assuming there was a reason.
        # The input to EnhancedPatchSelector is `patches` which has feature dimension C from `patchifier`.
        # C is `dimension` (the new parameter name for the old global DIMENSION).
        # So `EnhancedPatchSelector` should receive `patch_dim=dimension`.
        # The `self.feature_proj` layer in the original code projected from `DIMENSION` to `PATCH_SIZE * PATCH_SIZE * in_channels`.
        # This `self.patch_dim` was then passed to `EnhancedPatchSelector`.
        # This is a bit confusing. Let's re-evaluate:
        # 1. `patchifier` outputs `B, dimension, H, W`.
        # 2. `process_patches` reshapes to `B, H*W, dimension`.
        # 3. This is then fed to `self.feature_proj`.
        # 4. `self.feature_proj` projects from `dimension` to `calculated_patch_dim = patch_size * patch_size * in_channels`.
        # 5. This `calculated_patch_dim` is what `EnhancedPatchSelector` expects as `patch_dim`.

        self.calculated_patch_dim = patch_size * patch_size * in_channels # This is what selector expects as patch_dim
        self.feature_proj = nn.Linear(dimension, self.calculated_patch_dim) # Projects from CNN output dim to selector input dim

        # 设置top_k为patch总数的一半
        actual_top_k = self.num_patches // 2 if top_k == "half" else int(top_k) # Ensure top_k is int
        
        self.selector = EnhancedPatchSelector(
            patch_dim=self.calculated_patch_dim, # This is the dimension selector operates on
            num_heads=num_heads,                 # Use parameter: num_heads
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