import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageNetteDataset(Dataset):
    """ImageNette数据集加载器
    
    数据集说明:
    - ImageNet的10类子集
    - 包含训练集和验证集
    - 使用csv文件管理数据标签
    
    参数:
        root: 数据集根目录
        split: 数据集划分('train'或'val')
        transform: 数据预处理和增强操作
        
    数据格式:
    - 输入: RGB图像文件
    - 标签: 0-9的类别编号
    - 元信息: csv文件包含路径和标签
    
    使用方法:
    >>> dataset = ImageNetteDataset("data/", split="train")
    >>> image, label = dataset[0]
    """
    def __init__(self, root, split='train', transform=None):
        self.split = (split != 'train')
        self.root = root
        self.transform = transform
        self.images = pd.read_csv(
            root + '/noisy_imagenette.csv'
        )[['path', 'noisy_labels_0', 'is_valid']]
        self.images = self.images[self.images['is_valid'] == self.split]
        self.images['noisy_labels_0'] = pd.Categorical(
            self.images['noisy_labels_0']).codes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images.iloc[idx]
        img = Image.open(self.root + '/' + item['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['noisy_labels_0']
