import torch
import torch.nn as nn
import torch.nn.functional as F




class TwoLayerNN(nn.Module):
    """双层神经网络模块
    构建一个包含两个线性层和归一化层的神经网络模块
    主要用于特征变换和处理
    
    参数:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度,默认等于输入维度
        out_features: 输出特征维度,默认等于输入维度
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x


class SimplePatchifier(nn.Module):
    """图像分块处理器
    将输入图像划分为规则大小的图像块(patches)
    用于将图像转换为适合Vision GNN处理的patch序列
    
    参数:
        patch_size: 每个patch的边长大小,默认16
        
    输入:
        x: [B, C, H, W] 形状的图像张量
    输出:
        patches: [B, N, C, patch_size, patch_size] 形状的patch张量
        其中N是patch的数量
    """
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
            .unfold(2, self.patch_size, self.patch_size).contiguous()\
            .view(B, -1, C, self.patch_size, self.patch_size)
        return x

