import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import SimplePatchifier, TwoLayerNN

from patch_selector import ContentAwarePatchifier


class ViGBlock(nn.Module):
    """Vision GNN的核心构建模块
    实现了基于图神经网络的视觉特征提取和处理机制
    
    工作流程:
    1. 计算patch之间的相似度矩阵
    2. 基于相似度构建KNN图结构
    3. 通过图结构进行特征聚合和更新
    4. 使用多头注意力机制处理特征
    
    参数:
        in_features: 输入特征维度
        num_edges: 每个节点连接的边数量,控制图的稠密度
        head_num: 注意力头的数量,用于多头特征处理
    """
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        """前向传播
        1. 计算节点间相似度
        2. 构建KNN图结构
        3. 特征聚合与更新
        """
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(
            F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x


class VGNN(nn.Module):
    """Vision GNN网络主干
    将输入图像处理为patches并通过图神经网络进行特征提取
    
    架构组成:
    1. patch划分层: 将图像切分成固定大小的patches
    2. patch嵌入层: 将patches编码为高维特征向量
    3. 位置编码: 添加可学习的位置信息
    4. ViGBlock序列: 多层图神经网络处理单元
    
    参数:
        in_features: patch的原始特征维度(3*patch_size*patch_size)
        out_feature: 输出特征维度
        num_patches: patch的总数(H*W/patch_size^2)
        num_ViGBlocks: ViGBlock的层数
        num_edges: 每个节点的连接边数
        head_num: 注意力头数量
    """
    def __init__(self, in_features=3*16*16, out_feature=320, num_patches=196,
                 num_ViGBlocks=16, num_edges=9, head_num=1, patchifier_type='simple'):
        super().__init__()

        # 根据参数选择patchifier类型
        if patchifier_type == 'simple':
            self.patchifier = SimplePatchifier()
        else:  # patchifier_type == 'content_aware'
            self.patchifier = ContentAwarePatchifier(
                in_channels=3,
                top_k="half"
            )
            
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature//2),  # 恢复使用in_features
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        
        # 修改位置编码为动态创建
        self.pose_embedding = None
        self.out_feature = out_feature

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):
        # 注释说明patch选择过程
        # patchifier现在返回经过重要性筛选的patches
        x = self.patchifier(x)  # 现在只返回selected_patches
        B, N = x.shape[:2]  # 注意这里形状变化
        
        # 动态创建或调整位置编码大小
        if self.pose_embedding is None or self.pose_embedding.size(0) != N:
            self.pose_embedding = nn.Parameter(
                torch.rand(N, self.out_feature)).to(x.device)
        
        x = self.patch_embedding(x.reshape(B * N, -1)).reshape(B, N, -1)  # 使用reshape替代view
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class Classifier(nn.Module):
    """基于Vision GNN的图像分类器
    在VGNN主干网络基础上添加分类头进行图像分类
    
    结构:
    1. VGNN backbone: 提取图像的图结构特征
    2. 分类头: 将图特征映射到类别概率
    
    参数:
        n_classes: 分类类别数
        hidden_layer: 分类头中隐层的维度
        其他参数同VGNN
    
    输出:
        features: 图结构特征
        logits: 类别预测概率
    """
    def __init__(self, in_features=3*16*16, out_feature=320,
                 num_patches=196, num_ViGBlocks=16, hidden_layer=1024,
                 num_edges=9, head_num=1, n_classes=10, patchifier_type='simple'):
        super().__init__()
        
        self.backbone = VGNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num,
                             patchifier_type=patchifier_type)  # 传递patchifier类型参数
        
        # 注意：现在使用动态特征维度
        self.predictor = nn.Sequential(
            nn.LazyLinear(hidden_layer),  # 使用LazyLinear自动适应输入维度
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.reshape(B, -1))  # 使用reshape而不是view
        return features, x
