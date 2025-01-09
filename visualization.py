import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

class TrainingVisualizer:
    """训练过程可视化工具类
    
    功能：
    1. 损失曲线可视化
    2. 准确率曲线可视化
    3. 混淆矩阵绘制
    4. 模型参数分布分析
    5. 注意力权重可视化
    """
    
    def __init__(self, save_dir="./visualization"):
        """
        参数:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """绘制训练和验证的损失与准确率曲线
        
        参数:
            train_losses: 训练损失历史
            val_losses: 验证损失历史
            train_accs: 训练准确率历史
            val_accs: 验证准确率历史
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png")
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """绘制混淆矩阵
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/confusion_matrix.png")
        plt.close()
        
    def analyze_model_parameters(self, model):
        """分析模型参数分布
        
        参数:
            model: PyTorch模型实例
        """
        plt.figure(figsize=(12, 6))
        
        # 收集所有参数
        weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights.extend(param.data.cpu().numpy().flatten())
                
        # 绘制参数分布直方图
        plt.hist(weights, bins=50, density=True, alpha=0.7)
        plt.title('Model Parameters Distribution')
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/parameter_distribution.png")
        plt.close()
        
    def visualize_attention_weights(self, attention_weights, patch_size=16):
        """可视化注意力权重
        
        参数:
            attention_weights: 注意力权重矩阵 [num_heads, num_patches, num_patches]
            patch_size: 图像patch的大小
        """
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
        
        if num_heads == 1:
            axes = [axes]
            
        for head_idx, ax in enumerate(axes):
            sns.heatmap(attention_weights[head_idx], cmap='viridis', ax=ax)
            ax.set_title(f'Head {head_idx+1}')
            
        plt.suptitle('Attention Weights Visualization')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/attention_weights.png")
        plt.close()
        
    def plot_learning_rate_study(self, lrs, accuracies):
        """学习率分析图
        
        参数:
            lrs: 学习率列表
            accuracies: 对应的准确率列表
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, accuracies, '-o')
        plt.title('Learning Rate vs. Accuracy')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/lr_study.png")
        plt.close()
        
    def plot_patch_embeddings(self, embeddings):
        """可视化patch嵌入
        
        参数:
            embeddings: patch嵌入向量 [num_patches, embedding_dim]
        """
        # 使用t-SNE降维到2D
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title('Patch Embeddings Visualization (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(f"{self.save_dir}/patch_embeddings.png")
        plt.close()
