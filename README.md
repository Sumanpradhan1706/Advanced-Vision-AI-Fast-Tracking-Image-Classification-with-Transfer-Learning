# Fast-Tracking Image Classification with Transfer Learning

> Leveraging pre-trained deep learning models for efficient image classification on CIFAR-100 dataset

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)


## 📋 Overview

This project demonstrates the application of **transfer learning** using state-of-the-art pre-trained convolutional neural networks (CNNs) for image classification tasks. By leveraging powerful feature representations learned from the ImageNet dataset, we adapt three popular architectures—**ResNet50**, **VGG16**, and **MobileNetV2**—to classify images from the **CIFAR-100 dataset** (100 distinct classes).

Transfer learning significantly reduces computational requirements and training time compared to training models from scratch, while often achieving superior performance on smaller datasets.

## 🎯 Key Features

- **Multiple Pre-trained Architectures**: Comparison of ResNet50, VGG16, and MobileNetV2
- **Fine-tuning Strategy**: Selective layer unfreezing for optimal adaptation
- **Efficient Training**: Leverages pre-trained ImageNet weights to accelerate convergence
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Reproducible Workflow**: Clean, well-documented code structure

## 🏗️ Project Architecture

### Workflow

```
Data Loading & Preprocessing
         ↓
Model Preparation (Load pre-trained models)
         ↓
Layer Customization (Add classification heads)
         ↓
Fine-tuning (Selective layer unfreezing)
         ↓
Training & Validation
         ↓
Model Evaluation & Comparison
         ↓
Results Visualization & Model Persistence
```

## 📊 Dataset

**CIFAR-100 Dataset**
- **Classes**: 100 fine-grained categories
- **Training Samples**: 50,000 images
- **Test Samples**: 10,000 images
- **Image Resolution**: 32×32 pixels (resized to 224×224 for pre-trained models)

## 🧠 Models

### ResNet50
- **Depth**: 50 layers
- **Key Feature**: Residual connections for training very deep networks
- **Custom Head**: GlobalAveragePooling2D → Dense(1024, relu) → Dense(100, softmax)
- **Fine-tuning**: Last 30 layers unfrozen

### VGG16
- **Depth**: 16 layers
- **Key Feature**: Simple, sequential architecture with uniform 3×3 convolutions
- **Custom Head**: GlobalAveragePooling2D → Dense(512, relu) → Dense(100, softmax)
- **Fine-tuning**: Last 5 layers unfrozen

### MobileNetV2
- **Depth**: Lightweight architecture
- **Key Feature**: Depthwise separable convolutions for efficiency
- **Custom Head**: GlobalAveragePooling2D → Dense(256, relu) → Dense(100, softmax)
- **Fine-tuning**: Last 40 layers unfrozen

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.0+
- NumPy
- Matplotlib

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow numpy matplotlib
```

### Running the Project

1. **Navigate to the project directory**:
   ```bash
   cd anaconda_projects/db
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the notebook**:
   ```
   Fast-Tracking Image Classification with Transfer Learning.ipynb
   ```

4. **Execute cells sequentially**:
   - Cell 1: Data Loading and Preprocessing
   - Cell 2: Model Preparation
   - Cell 3: Fine-tuning and Training
   - Cell 4: Evaluation and Visualization

## 📈 Workflow Details

### 1. Data Loading and Preprocessing
- Loads CIFAR-100 dataset using Keras utilities
- Resizes images from 32×32 to 224×224 pixels (required by pre-trained models)
- Applies model-specific preprocessing:
  - ResNet50: Normalization for ImageNet-trained weights
  - VGG16: Channel-wise mean subtraction
  - MobileNetV2: Channel-wise normalization to [-1, 1]

### 2. Model Preparation
- Loads pre-trained ImageNet weights without top classification layers
- Replaces top layers with custom dense layers suitable for 100 classes
- Freezes base model layers to preserve learned feature representations
- Compiles models with:
  - **Optimizer**: Adam
  - **Loss**: Sparse Categorical Crossentropy
  - **Metric**: Accuracy

### 3. Fine-tuning and Training
- Unfreezes a portion of top layers to adapt to CIFAR-100 features
- Trains models for configurable epochs (default: 3)
- Monitors performance on validation set during training
- Generates training history for visualization

### 4. Model Evaluation
- Evaluates each model on held-out test set
- Compares accuracy metrics across architectures
- Generates training/validation curves for loss and accuracy

### 5. Results Persistence
- Saves trained models in HDF5 format:
  - `resnet50_cifar100.h5`
  - `vgg16_cifar100.h5`
  - `mobilenetv2_cifar100.h5`

## 📊 Expected Results

The notebook evaluates all three models and provides:
- **Test Accuracy**: Performance on held-out test set
- **Training Curves**: Visualization of loss and accuracy over epochs
- **Model Comparison**: Side-by-side accuracy metrics

Example output:
```
ResNet50 Accuracy: 0.XX
VGG16 Accuracy: 0.XX
MobileNetV2 Accuracy: 0.XX
```

## 🔧 Customization

### Adjust Training Parameters
```python
epochs = 5  # Increase for better convergence
batch_size = 32  # Modify batch size
```

### Modify Fine-tuning Strategy
```python
# Unfreeze different number of layers
for layer in model_resnet50.layers[:-50]:  # Unfreeze more layers
    layer.trainable = False
```

### Change Dense Layer Sizes
```python
# Customize hidden layer dimensions
x = Dense(2048, activation='relu')(x)  # Increase capacity
```

## 📁 Project Structure

```
Advanced Vision AI Fast-Tracking Image Classification with Transfer Learning/
├── README.md
└── anaconda_projects/
    └── db/
        └── Fast-Tracking Image Classification with Transfer Learning.ipynb
```

## 🎓 Learning Objectives

By completing this project, you will understand:
- ✅ Fundamentals of transfer learning
- ✅ How to leverage pre-trained models for new tasks
- ✅ Fine-tuning strategies for domain adaptation
- ✅ Comparison of different CNN architectures
- ✅ Model evaluation and visualization techniques
- ✅ Best practices for efficient deep learning workflows

## 💡 Key Concepts

| Concept | Description |
|---------|-------------|
| **Transfer Learning** | Reusing learned features from large datasets to solve new tasks |
| **Feature Extraction** | Using pre-trained layers as fixed feature detectors |
| **Fine-tuning** | Adapting pre-trained weights to new data by unfreezing top layers |
| **ImageNet** | Large-scale dataset (1.2M images, 1000 classes) used for pre-training |
| **CIFAR-100** | Smaller dataset (60K images, 100 classes) for downstream task |

## 🔍 Comparison of Architectures

| Model | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| ResNet50 | 25.5M | Medium | High | Balanced performance |
| VGG16 | 138M | Slow | Good | Feature visualization |
| MobileNetV2 | 3.5M | Fast | Very Good | Mobile/edge devices |

## 📝 Dependencies

- **TensorFlow**: Deep learning framework for model building and training
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Visualization of training curves and results

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Add new model architectures
- Enhance documentation

## 📚 References

- [ImageNet Paper](https://www.image-net.org/)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [VGG: Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
- [MobileNetV2: Efficient CNNs for Mobile Vision](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/guide/transfer_learning)

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created as part of the Advanced Vision AI learning series focusing on practical deep learning applications.

## 🙏 Acknowledgments

- TensorFlow/Keras team for excellent deep learning tools
- ImageNet and CIFAR-100 dataset creators
- Open-source community for continuous support

---

**Last Updated**: December 2025

**Status**: Active Development

For questions or issues, please open an issue or contact the repository maintainer.
