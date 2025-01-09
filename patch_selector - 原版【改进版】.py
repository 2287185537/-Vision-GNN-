import configparser
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
    """基于内容重要性的patch选择模块
    
    该模块通过可学习的多头注意力机制对输入的patches进行重要性评分和筛选。
    增加了残差连接和LayerNorm来提升性能。
    
    参数说明:
        patch_dim (int): patch的特征维度，默认从配置文件读取
        num_heads (int): 注意力头的数量，多头可以关注不同特征模式
        top_k (int, optional): 需要选择的patch数量，默认选择一半
    """
    def __init__(self, patch_dim=DIMENSION, num_heads=HEAD_NUM, top_k=None):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = patch_dim // num_heads
        
        # 改进多头注意力网络结构
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_dim, self.head_dim * 2),  # 增加中间层维度
                nn.LayerNorm(self.head_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),  # 添加dropout
                nn.Linear(self.head_dim * 2, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.GELU(),
                nn.Linear(self.head_dim, 1)
            ) for _ in range(num_heads)
        ])
        
        # 添加特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(num_heads, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(patch_dim)
        self.top_k = top_k
        
    def forward(self, patches):
        """前向传播函数
        
        参数:
            patches (torch.Tensor): 形状为[B, N, D]的输入特征张量
                B: batch size
                N: patch数量
                D: 每个patch的特征维度
                
        返回:
            selected_patches (torch.Tensor): 选择后的patches，形状为[B, K, D]
            attention_weights (torch.Tensor): 注意力权重，形状为[B, N]
        """
        B, N, D = patches.shape
        
        # 应用层归一化
        normalized_patches = self.layer_norm(patches)
        
        # 计算多头注意力权重
        attention_weights = []
        for head in self.attention:
            scores = head(normalized_patches.reshape(-1, D)).reshape(B, N)
            attention_weights.append(scores)
        
        # 将所有头的注意力分数堆叠并融合
        stacked_attention = torch.stack(attention_weights, dim=-1)  # [B, N, H]
        fused_attention = self.fusion(stacked_attention).squeeze(-1)  # [B, N]
        
        # 添加残差连接
        attention_weights = F.softmax(fused_attention, dim=1)
        
        # 选择top-k个patches
        k = self.top_k if self.top_k is not None else (N // 2)
        _, indices = torch.topk(attention_weights, k, dim=1)
        selected_patches = torch.gather(patches, 1,
            indices.unsqueeze(-1).expand(-1, -1, D))
            
        return selected_patches, attention_weights

class ImportanceScorer(nn.Module):
    """改进的Patch重要性评分模块
    
    通过多层神经网络学习patch的重要性特征，结合位置信息进行评分。
    
    关键组件:
        - importance_net: 多层打分网络，包含多次降维和归一化
        - pos_embedding: 可学习的位置编码，帮助模型理解空间关系
        - pos_dropout: 位置编码的dropout，防止过拟合
        
    网络结构说明:
        1. 输入层 -> patch_dim维
        2. 第一隐层 -> patch_dim/2维 (带归一化和dropout)
        3. 第二隐层 -> patch_dim/4维
        4. 输出层 -> 1维重要性分数
    """
    def __init__(self, patch_dim=DIMENSION):
        super().__init__()
        
        # 更深层的网络结构
        self.importance_net = nn.Sequential(
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(patch_dim, patch_dim//2),
            nn.LayerNorm(patch_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(patch_dim//2, patch_dim//4),
            nn.LayerNorm(patch_dim//4),
            nn.GELU(),
            nn.Linear(patch_dim//4, 1),
            nn.Sigmoid()
        )
        
        # 改进的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, patch_dim))
        self.pos_dropout = nn.Dropout(0.1)
        
    def forward(self, patches):
        # 添加位置信息并应用dropout
        patches = patches + self.pos_dropout(self.pos_embedding)
        
        # 计算重要性分数
        scores = self.importance_net(patches).squeeze(-1)
        return scores

class EnhancedPatchSelector(PatchSelector):
    """增强版patch选择器
    
    在基础PatchSelector上增加了以下增强功能：
    1. 重要性评分：使用ImportanceScorer评估patch重要性
    2. 特征增强：使用多层神经网络增强选中patch的特征
    3. 注意力融合：结合多头注意力和重要性分数
    4. 残差连接：保持原始特征信息
    
    特征增强器结构:
        1. 输入维度 -> 4倍维度扩展
        2. 4倍维度 -> 2倍维度
        3. 2倍维度 -> 原始维度
        每层都包含LayerNorm和Dropout
        
    注意力融合说明:
        将多头注意力scores和重要性scores通过线性层融合，
        得到更准确的patch选择依据
    """
    def __init__(self, patch_dim=DIMENSION, num_heads=HEAD_NUM, top_k=None):
        super().__init__(patch_dim, num_heads, top_k)
        
        self.importance_scorer = ImportanceScorer(patch_dim)
        
        # 改进的特征增强器
        self.feature_enhancer = nn.Sequential(
            nn.Linear(patch_dim, patch_dim * 4),
            nn.LayerNorm(patch_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(patch_dim * 4, patch_dim * 2),
            nn.LayerNorm(patch_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(patch_dim * 2, patch_dim),
            nn.LayerNorm(patch_dim)
        )
        
        # 添加注意力融合层
        self.attention_fusion = nn.Sequential(
            nn.Linear(2, 1),  # 融合两种注意力分数
            nn.Sigmoid()
        )
        
    def forward(self, patches):
        B, N, D = patches.shape
        
        # 计算内容重要性分数
        importance_scores = self.importance_scorer(patches)
        
        # 计算注意力分数
        attention_weights = []
        for head in self.attention:
            scores = head(patches.reshape(-1, D)).reshape(B, N)
            attention_weights.append(scores)
        attention_weights = torch.stack(attention_weights, dim=1)
        
        # 融合多头注意力
        avg_attention = attention_weights.mean(dim=1)
        
        # 组合两种分数
        combined_scores = torch.stack([avg_attention, importance_scores], dim=-1)
        final_scores = self.attention_fusion(combined_scores).squeeze(-1)
        final_scores = F.softmax(final_scores, dim=1)
        
        # 选择patches
        k = self.top_k if self.top_k is not None else (N // 2)
        _, indices = torch.topk(final_scores, k, dim=1)
        
        # 选择并增强特征
        selected_patches = torch.gather(patches, 1,
            indices.unsqueeze(-1).expand(-1, -1, D))
        enhanced_patches = self.feature_enhancer(selected_patches)
        
        # 添加残差连接
        enhanced_patches = enhanced_patches + selected_patches
        
        return enhanced_patches, final_scores

# 更新ContentAwarePatchifier使用增强版选择器
class ContentAwarePatchifier(nn.Module):
    """具有内容感知筛选功能的patch处理器
    
    该模块首先将输入图像划分为patches并进行特征提取，
    然后使用PatchSelector模块选择最重要的patches。
    
    参数:
        in_channels: 输入图像的通道数
        top_k: 需要选择保留的patch数量，默认为总patches数量的一半
    """
    def __init__(self, in_channels=3, top_k="half"):
        super().__init__()
        
        # 修改patchifier以输出正确的特征维度
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channels, DIMENSION,
                     kernel_size=PATCH_SIZE,
                     stride=PATCH_SIZE),
            nn.BatchNorm2d(DIMENSION),
            nn.GELU()
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
        
        # patch划分和特征提取
        patches = self.patchifier(x)
        B, C, H, W = patches.shape
        
        # 重排张量维度并投影到正确的维度
        patches = patches.permute(0, 2, 3, 1).reshape(B, H*W, C)
        patches = self.feature_proj(patches)  # 投影到期望的维度
        
        # 使用selector选择重要的patches
        selected_patches, _ = self.selector(patches)
        
        return selected_patches