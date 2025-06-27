# EasyMLLM

[English](README.md) | [中文](README_zh.md)

EasyMLLM 是一个适用于初学者的机器学习库，旨在帮助像我一样入门机器学习的同学进行快速的上手以及提供简单的API进行各种模型和数据集的训练、测试和推理。它支持模块化和可扩展性，适用于不同的机器学习任务。同时将持续扩展不同领域，例如针对模型的攻击，以及模型融合等前沿技术的实现以及易复现的代码。

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/Xxxxsir/EasyMLLM.git
   cd EasyMLLM
   ```

2. 创建虚拟环境（推荐）：
   ```bash
   conda env create -f environment.yml
   conda activate easymllm
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 下载数据集（如 MNIST 或 COVID Radiography），并将其放置在 `data` 目录下。

## 支持的模型

- **LeNet**
- **ViT (Vision Transformer)**

## 支持的数据集

- **MNIST**
- **COVID Radiography**

## 使用方法

### 训练

使用 `train.py` 脚本进行模型训练。示例：

```bash
python train.py --model_name lenet \
    --dataset_name mnist \
    --classes_num 10 \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --pre_trained False 
```

### 推理

使用 `inference.py` 脚本对单张图片或批量图片进行推理。示例：

#### 单张图片推理
```bash
python inference.py --model_name lenet --checkpoint_path ./model_ckpt/mnist_lenet.pth --test_data_path ./data/mnist/test/0.png --dataset mnist --num_classes 10
```

#### 批量图片推理
```bash
python inference.py --model_name lenet --checkpoint_path ./model_ckpt/mnist_lenet.pth --test_data_path ./data/mnist/test --dataset mnist --num_classes 10
```

## 目录结构

```
EasyMLLM/
├── attack/                # 对抗攻击脚本
├── data/                  # 数据集目录
├── models/                # 模型定义
├── model_ckpt/            # 模型检查点
├── utils.py               # 工具函数
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── datasets.py            # 数据集加载与预处理
└── README.md              # 项目文档
```

## 许可证

本项目基于 MIT 许可证。详情请参阅 LICENSE 文件。
