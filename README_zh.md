# Vision GNN (ViG) 模型复现

本项目旨在复现论文 ["Vision GNN: An Image is Worth Graph of Nodes"](https://arxiv.org/abs/2206.00272) 中提出的 Vision Graph Neural Network (ViG) 模型。

原始英文版README请见 [README.md](README.md)。
更详细的复现报告（英文）请见 [docs/main.pdf](docs/main.pdf)。

## 目录
- [项目简介](#项目简介)
- [环境设置](#环境设置)
- [数据集](#数据集)
- [配置文件说明](#配置文件说明)
- [模型训练](#模型训练)
- [模型架构](#模型架构)
- [主要文件说明](#主要文件说明)
- [结果与可视化](#结果与可视化)
- [可选：如何贡献](#可选如何贡献)
- [可选：许可证](#可选许可证)

## 项目简介
本项目提供了一个基于PyTorch的Vision GNN (ViG)图像分类模型的实现。ViG模型将图像转换为节点图（Graph of Nodes），并利用图神经网络进行特征提取和分类。

## 环境设置
1.  **Python版本:** 建议使用 Python 3.8 或更高版本。
2.  **创建虚拟环境 (可选但推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **安装依赖:**
    项目所需的依赖项记录在 `requirements.txt` 文件中。通过以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括 `torch`, `torchvision`, `pandas`, `Pillow`, `tqdm`, `matplotlib`, `seaborn`, `scikit-learn`。

## 数据集
本项目使用 **ImageNette** 数据集，它是ImageNet的一个10类小型子集，常用于快速实验。

1.  **下载数据集:**
    通过运行 `get_dataset.py` 脚本来下载和解压数据集：
    ```bash
    python get_dataset.py
    ```
    该脚本会自动下载 `imagenette2-320.tgz` (约1.5GB) 并将其解压到 `data/` 目录下。
2.  **数据集路径:**
    脚本执行完毕后，数据集应位于 `./data/imagenette2-320/`。
    配置文件 `confs/main.ini` 中的 `PATH` 参数（默认为 `/workspace/复现/VisionGNN-源码/VisionGNN/data/imagenette2-320`）需要指向此路径。请根据实际情况修改此配置。
3.  **数据加载:**
    `dataset.py` 中的 `ImageNetteDataset` 类负责加载数据，并使用 `data/noisy_imagenette.csv` 文件管理图像路径和标签。
    训练时会进行数据增强，包括随机裁剪、RandAugment、随机水平翻转和随机擦除等。

## 配置文件说明
项目的主要配置通过 `confs/main.ini` 文件进行管理。该文件包含以下几个部分：

*   **`[TRAIN]` 部分:** 训练相关参数
    *   `BATCH_SIZE`: 每个训练批次的图像数量 (例如: `32`)。
    *   `LR`: 学习率 (在 `train.py` 中当前硬编码为 `0.0001`，原配置为 `0.00001`)。
    *   `EPOCHS`: 训练的总轮数 (例如: `80`)。
    *   `SAVE_DIR`: 训练日志和模型保存的目录 (例如: `train_log/`)。

*   **`[DATASET]` 部分:** 数据集相关参数
    *   `PATH`: 数据集根目录的路径。
    *   `NUM_CLASSES`: 数据集的类别数量 (ImageNette为 `10`)。

*   **`[MODEL]` 部分:** 模型架构相关参数
    *   `PATCH_SIZE`: 图像块 (patch) 的边长大小 (例如: `16`)。
    *   `DIMENSION`: 模型的主要特征维度 (例如: `192`)。此维度也用作ViG模块的输出特征维度，以及内容感知Patch选择器内部CNN的维度。
    *   `DEPTH`: 模型中 `ViGBlock` 的层数 (例如: `16`)。
    *   `NUM_EDGES`: 在 `ViGBlock` 中构建KNN图时，每个节点连接的边数 (例如: `9`)。
    *   `HEAD_NUM`: `ViGBlock` 中多头注意力机制的头数，也用于Patch选择器中的多头注意力 (例如: `2`)。
    *   `PATCHIFIER_TYPE` (可选): Patch选择器的类型。可设置为 `'simple'` 或 `'content_aware'`。默认为 `'content_aware'` (在 `train.py` 中设置)。

## 模型训练
1.  **确保配置正确:** 检查 `confs/main.ini` 中的参数，特别是数据集路径。
2.  **开始训练:**
    运行 `train.py` 脚本开始模型训练：
    ```bash
    python train.py
    ```
3.  **训练过程:**
    *   训练脚本会使用 `Adam` 优化器、`CosineAnnealingLR` 学习率调度器和 `GradScaler` 进行混合精度训练。
    *   训练日志（包括损失、训练集准确率、验证集准确率）会输出到控制台，并保存在 `SAVE_DIR` 下的带时间戳的子目录中 (例如 `train_log/YYYYMMDD_HHMM/train.log`)。
    *   验证集上表现最佳的模型权重会保存为 `best_model.pth`。
    *   每10个epoch会保存一次当时的模型权重。

## 模型架构
模型主要由 `Classifier` 类（在 `model.py` 中定义）实现，它包含一个 `VGNN` 主干网络和一个分类头。

*   **`VGNN` (Vision GNN):**
    *   **Patch划分与选择:** 首先，输入图像被处理成一系列图像块 (patches)。
        *   `SimplePatchifier` (`utility.py`): 一种简单的将图像分割成固定大小块的方法。
        *   `ContentAwarePatchifier` (`patch_selector.py`): 一种更复杂的机制，它使用 `EnhancedPatchSelector` 来选择内容感知的重要patches。`EnhancedPatchSelector` 结合了基础的注意力分数和通过 `ImportanceScorer` 计算的重要性分数。
    *   **Patch嵌入:** 选定的patches被展平并通过一个线性层序列进行嵌入。
    *   **位置编码:** 为patch嵌入添加可学习的位置信息。
    *   **`ViGBlock` 序列:** 核心的图神经网络模块。
        *   **图构建:** 在每个 `ViGBlock` 中，根据patch特征间的相似度构建一个K近邻 (KNN) 图。
        *   **特征聚合与更新:** 通过图结构聚合邻居节点的特征，并利用多头注意力机制（`nn.Conv1d` 实现）和MLP（`TwoLayerNN`）更新节点特征。
*   **分类头:**
    *   `VGNN` 的输出特征（所有patch特征的扁平化）被送入一个包含 `LazyLinear` 层的分类头，以预测最终的类别。

## 主要文件说明
*   `train.py`: 包含完整的训练和验证流程。
*   `model.py`: 定义了核心模型架构，包括 `Classifier`, `VGNN`, 和 `ViGBlock`。
*   `dataset.py`: 包含了 `ImageNetteDataset` 类，用于加载和预处理ImageNette数据集。
*   `patch_selector.py`: 实现了内容感知的Patch选择机制，包括 `ContentAwarePatchifier`, `EnhancedPatchSelector`, `PatchSelector`, 和 `ImportanceScorer`。
*   `utility.py`: 提供辅助模块，如 `SimplePatchifier` 和 `TwoLayerNN`。
*   `visualization.py`: 包含 `TrainingVisualizer` 类，用于在训练过程中生成和保存可视化图表（如损失曲线、混淆矩阵）。
*   `get_dataset.py`: 用于自动下载和解压ImageNette数据集的脚本。
*   `confs/main.ini`: 项目的配置文件，用于设置训练、数据集和模型参数。
*   `requirements.txt`: 列出了项目所需的Python依赖包。
*   `README.md`: 项目的英文版说明文档。
*   `docs/main.pdf`: 详细的英文复现报告。

## 结果与可视化
*   **日志:** 训练过程中的详细日志（损失、准确率等）保存在 `SAVE_DIR`（默认为 `train_log/`）下的时间戳子目录中的 `train.log` 文件。
*   **最佳模型:** 验证集上性能最佳的模型权重保存在同一日志子目录下的 `best_model.pth` 文件。
*   **定期模型保存:** 每10个epoch的模型权重也会保存在该子目录下。
*   **可视化图表:** `TrainingVisualizer` (`visualization.py`) 会在训练过程中生成图表，并默认保存在 `visualization_results/` 目录下。这些图表包括：
    *   `training_curves.png`: 训练和验证的损失及准确率曲线。
    *   `confusion_matrix.png`: 在验证集上绘制的混淆矩阵（每10个epoch更新）。
    *   `parameter_distribution.png`: 模型参数权重的分布直方图（每10个epoch更新）。

## (可选) 如何贡献
欢迎对本项目进行改进和贡献！如果您有新的想法或发现了问题，请遵循以下步骤：
1.  Fork本仓库。
2.  创建一个新的分支 (`git checkout -b feature/YourFeature` 或 `bugfix/YourBug`)。
3.  进行修改并确保代码清晰、包含注释和必要的测试。
4.  提交您的更改 (`git commit -m 'Add some feature'`)。
5.  将您的分支推送到GitHub (`git push origin feature/YourFeature`)。
6.  创建一个Pull Request供我们审查。

## (可选) 许可证
本项目采用 [MIT 许可证](LICENSE) (如果 `LICENSE` 文件存在并使用MIT)。请查看 `LICENSE` 文件获取详细信息。
