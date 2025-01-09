import time
import logging
import configparser
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier
from dataset import ImageNetteDataset
from visualization import TrainingVisualizer


def load_dataset(path, batch_size):
    """加载数据集并构建数据加载器
    
    实现了多种数据增强策略:
    - 随机裁剪和调整大小
    - RandAugment增强
    - 随机水平翻转
    - 标准化处理
    - 随机擦除
    
    参数:
        path: 数据集路径
        batch_size: 批次大小
        
    返回:
        train_dataloader: 训练集加载器
        val_dataloader: 验证集加载器
    """
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    # is the order of tranfoms important? Is this the best order?
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
    ])

    # cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    # mixup = v2.MixUp(num_classes=NUM_CLASSES)
    # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    train_dataset = ImageNetteDataset(
        path, split='train', transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_dataset = ImageNetteDataset(path, split='val', transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader


def train_step(model, dataloader, optimizer, criterion, device, epoch=None, mix_aug=None):
    """执行一个训练epoch
    
    工作流程:
    1. 将模型设为训练模式
    2. 批次迭代训练数据
    3. 前向传播计算损失
    4. 反向传播更新参数
    5. 记录loss和准确率
    
    参数:
        model: 待训练的模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前epoch数
        mix_aug: 混合增强策略
        
    返回:
        epoch平均loss和准确率
    """
    running_loss, correct, total = [], 0, 0
    model.train()

    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device).long()

        if mix_aug is not None:
            x, y = mix_aug(x, y)

        optimizer.zero_grad()

        _, pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # predicted_class = pred.argmax(dim=1, keepdim=False)

        # total += y.numel()
        # correct += (predicted_class == y).sum().item()

        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
    # acc = correct / total
    acc = None
    return sum(running_loss) / len(running_loss), acc


def validation_step(model, dataloader, device):
    """模型验证，并收集预测结果
    
    返回:
        acc: 验证集准确率
        all_preds: 所有预测结果
        all_targets: 所有真实标签
    """
    correct, total = 0, 0
    model.eval()
    
    all_preds = []
    all_targets = []
    
    validation_bar = tqdm(dataloader)
    with torch.no_grad():
        for x, y in validation_bar:
            x, y = x.to(device), y.to(device)
            
            _, pred = model(x)
            predicted_class = pred.argmax(dim=1, keepdim=False)
            
            # 收集预测结果和真实标签
            all_preds.extend(predicted_class.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            total += y.numel()
            correct += (predicted_class == y).sum().item()
            
            acc = correct / total
            validation_bar.set_description(
                f'accuracy is {round(acc*100, 2)}% until now.')
    
    return acc, all_preds, all_targets


def train(conf, device):
    """完整训练流程
    
    主要步骤:
    1. 配置日志和保存目录
    2. 加载和初始化模型
    3. 准备数据集和优化器
    4. 循环训练指定epochs
    5. 保存最佳模型检查点
    
    参数:
        conf: 配置参数字典
        device: 计算设备
        
    训练过程中会记录:
    - 训练loss历史
    - 训练/验证准确率
    - 最佳模型权重
    - 训练日志
    """
    save_dir = Path(conf['TRAIN']['SAVE_DIR']) / \
        datetime.now().strftime('%Y%m%d_%H%M')
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(save_dir / 'train.log'),
        logging.StreamHandler()
    ])

    model = Classifier(n_classes=conf['DATASET'].getint('NUM_CLASSES'),
                       num_ViGBlocks=conf['MODEL'].getint('DEPTH'),
                       out_feature=conf['MODEL'].getint('DIMENSION'),
                       num_edges=conf['MODEL'].getint('NUM_EDGES'),
                       head_num=conf['MODEL'].getint('HEAD_NUM'))
    model.to(device)
    logging.info('Model loaded')
    logging.info({section: dict(conf[section]) for section in conf.sections()})

    train_dataloader, val_dataloader = load_dataset(
        conf['DATASET']['PATH'], conf['TRAIN'].getint('BATCH_SIZE'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf['TRAIN'].getfloat('LR'))

    loss_hisroty, train_acc_hist, val_acc_hist = [], [], []
    max_val_acc = 0

    # 初始化可视化工具
    visualizer = TrainingVisualizer(save_dir="visualization_results")

    since = time.time()
    for epoch in range(1, conf['TRAIN'].getint('EPOCHS')+1):

        loss, train_acc = train_step(
            model, train_dataloader, optimizer, criterion, device, epoch)
        # 修改这里以接收预测结果
        val_acc, epoch_preds, epoch_targets = validation_step(model, val_dataloader, device)

        loss_hisroty.append(loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        logging.info(f'Epoch: {epoch}, Loss: {loss}, Val acc: {val_acc*100}')

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), save_dir /
                       f'best_model.pth')

        # 可视化当前训练状态
        visualizer.plot_training_curves(loss_hisroty, val_acc_hist, 
                                      train_acc_hist, val_acc_hist)
        
        # 每10个epoch保存一次完整的分析结果
        if (epoch + 1) % 10 == 0:
            # 使用当前epoch的预测结果绘制混淆矩阵
            visualizer.plot_confusion_matrix(
                epoch_targets, epoch_preds,
                class_names=[str(i) for i in range(10)]
            )
            
            # 模型参数分析
            visualizer.analyze_model_parameters(model)
            
            # 保存模型
            torch.save(model.state_dict(), 
                      f"{conf['TRAIN']['SAVE_DIR']}/model_epoch_{epoch+1}.pth")

    logging.info('Training Finished.')
    logging.info(f'Max validation accuracy is {round(max_val_acc*100, 2)}%')
    logging.info(f'elapsed time is {time.time() - since}')


def validate_config(conf):
    """验证配置文件的正确性"""
    required_sections = ['TRAIN', 'DATASET', 'MODEL']
    required_options = {
        'TRAIN': ['BATCH_SIZE', 'LR', 'EPOCHS', 'SAVE_DIR'],
        'DATASET': ['PATH', 'NUM_CLASSES'],
        'MODEL': ['PATCH_SIZE', 'DIMENSION', 'DEPTH', 'NUM_EDGES', 'HEAD_NUM']
    }
    
    # 检查必需的配置sections
    for section in required_sections:
        if section not in conf:
            raise ValueError(f"Missing required section: {section}")
            
    # 检查每个section中必需的配置项
    for section, options in required_options.items():
        for option in options:
            if option not in conf[section]:
                raise ValueError(f"Missing required option: {option} in section {section}")
            
    # 验证数值的合理性
    try:
        batch_size = conf['TRAIN'].getint('BATCH_SIZE')
        if batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
            
        lr = conf['TRAIN'].getfloat('LR')
        if lr <= 0:
            raise ValueError("LR must be positive")
            
        epochs = conf['TRAIN'].getint('EPOCHS')
        if epochs <= 0:
            raise ValueError("EPOCHS must be positive")
            
        num_classes = conf['DATASET'].getint('NUM_CLASSES')
        if num_classes <= 0:
            raise ValueError("NUM_CLASSES must be positive")
    except ValueError as e:
        raise ValueError(f"Invalid configuration value: {str(e)}")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf = configparser.ConfigParser()
    conf.read('/workspace/复现/VisionGNN-源码/VisionGNN/confs/main.ini')
    
    # 添加配置验证
    validate_config(conf)
    
    train(conf, device)
