# EasyMLLM

[English](README.md) | [中文](README_zh.md)

EasyMLLM is a beginner-friendly machine learning library designed to help students like myself quickly get started with machine learning by providing simple APIs for training, testing, and inference across various models and datasets. It supports modularity and extensibility, making it suitable for different machine learning tasks. The library will continue to expand into various fields, including implementing and reproducing cutting-edge techniques such as model attacks and model merging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Xxxxsir/EasyMLLM.git
   cd EasyMLLM
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   conda env create -f environment.yml
   conda activate easymllm
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the datasets (e.g., MNIST or COVID Radiography) and place them in the `data` directory.

## Supported Models

- **LeNet**
- **ViT (Vision Transformer)**

## Supported Datasets

- **MNIST**
- **COVID Radiography**

## Usage

### Training

To train a model, use the `train.py` script. Example:

```bash
python train.py --model_name lenet \
    --dataset_name mnist \
    --classes_num 10 \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --pre_trained False 
```

### Inference

To perform inference on a single image or a batch of images, use the `inference.py` script. Example:

#### Single Image Inference
```bash
python inference.py --model_name lenet --checkpoint_path ./model_ckpt/mnist_lenet.pth --test_data_path ./data/mnist/test/0.png --dataset mnist --num_classes 10
```

#### Batch Inference
```bash
python inference.py --model_name lenet --checkpoint_path ./model_ckpt/mnist_lenet.pth --test_data_path ./data/mnist/test --dataset mnist --num_classes 10
```

## Directory Structure

```
EasyMLLM/
├── attack/                # Adversarial attack scripts
├── data/                  # Dataset directory
├── models/                # Model definitions
├── model_ckpt/            # Model checkpoints
├── utils.py               # Utility functions
├── train.py               # Training script
├── inference.py           # Inference script
├── datasets.py            # Dataset loading and preprocessing
└── README.md              # Project documentation
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

