import os
import requests
from pathlib import Path
import tarfile

def download_dataset():
    """下载和解压ImageNette数据集
    
    执行步骤:
    1. 创建数据目录
    2. 从S3下载数据集压缩包
    3. 解压文件到指定目录
    
    数据集信息:
    - 名称: imagenette2-320
    - 来源: fast.ai
    - 大小: 约1.5GB
    - 格式: tar.gz压缩包
    
    注意事项:
    - 需要稳定的网络连接
    - 确保足够的磁盘空间
    - 解压过程可能需要几分钟
    """
    # 创建data目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 下载数据集
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = data_dir / "imagenette2-320.tgz"
    
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    # 解压文件
    print("Extracting files...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=data_dir)
    
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    download_dataset()