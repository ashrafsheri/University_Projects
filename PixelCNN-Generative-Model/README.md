# PixelCNN: Autoregressive Image Generation

A PyTorch implementation of PixelCNN for generating images pixel-by-pixel using autoregressive modeling on the MNIST dataset.

## 🎯 Project Overview

This project implements PixelCNN, a generative model that learns to generate images by modeling the joint distribution of pixels autoregressively. The model predicts each pixel conditioned on all previously generated pixels, creating realistic handwritten digit images.

## 🧠 Key Concepts

### Autoregressive Image Generation

PixelCNN models images as a sequence of pixels, where each pixel is generated conditionally on all previous pixels:

```
p(x) = ∏ p(x_i | x_1, x_2, ..., x_{i-1})
```

This is achieved through:
- **Masked Convolutions**: Ensure each pixel only sees previous pixels
- **Residual Blocks**: Capture complex pixel dependencies
- **Sequential Generation**: Generate images pixel-by-pixel from top-left to bottom-right

## 🏗️ Architecture Components

### 1. Masked Convolutions

Two types of masks enforce the autoregressive property:

- **Mask Type A** (first layer):
  - Current pixel **cannot** see itself
  - Sees only strictly previous pixels
  
- **Mask Type B** (subsequent layers):
  - Current pixel **can** see itself
  - Maintains causality through the network

```python
class PixelConv(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, ...):
        # Creates masked convolution with type 'A' or 'B'
```

### 2. Residual Blocks

Stacked residual blocks improve gradient flow and model capacity:

```python
class ResidualBlock(nn.Module):
    # ReLU → Conv1x1 → ReLU → MaskedConv → ReLU → Conv1x1
    # Skip connection: input + output
```

### 3. PixelCNN Architecture

```
Input (28×28) 
    ↓
Masked Conv 7×7 (Type A) → 64 channels
    ↓
4× Residual Blocks
    ↓
ReLU → Masked Conv 1×1 (Type B) → ReLU → Conv 1×1
    ↓
Output logits (28×28)
```

## 📁 Project Structure

```
PixelCNN-Generative-Model/
├── README.md                          # This file
├── 26100342_PA3_1.ipynb              # Part 1: PixelCNN Implementation (PyTorch)
├── 26100342_PA3_1.py                 # Python script version
├── 26100342_PA3_2.ipynb              # Part 2: Gated PixelCNN
├── 26100342_PA3_2.py                 # Python script version  
├── 26100342_PA3_3.ipynb              # Part 3: Advanced Topics
└── 26100342_PA3_3.py                 # Python script version
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Training the Model

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Binarize images
train_images = (train_data.data > 127.5).float()

# Initialize model
model = PixelCNN(input_channels=1, hidden_channels=64, num_residual_blocks=4)

# Train with Binary Cross-Entropy
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (30 epochs recommended)
```

### Generating Images

```python
@torch.no_grad()
def generate_images(model, num_images=4):
    model.eval()
    samples = torch.zeros((num_images, 1, 28, 28))
    
    # Generate pixel-by-pixel
    for i in range(28):
        for j in range(28):
            logits = model(samples)
            probs = torch.sigmoid(logits[:, :, i, j])
            samples[:, :, i, j] = torch.bernoulli(probs)
    
    return samples
```

## 📊 Experiments & Results

### Training Progression

The model was trained for different epoch counts to observe learning:

| Epochs | Train Loss | Val Loss | Image Quality |
|--------|-----------|----------|---------------|
| 10     | ~0.15     | ~0.14    | Blurry, lacks structure |
| 20     | ~0.10     | ~0.12    | Some structure visible |
| 30     | ~0.08     | ~0.11    | Clear digits, well-formed |

### Observations

1. **Epoch 10**: Images are very blurry with no clear digit structure
2. **Epoch 20**: Loss reduced but digits still unclear; basic structure emerging
3. **Epoch 30**: Significant improvement with clear, recognizable digits

## 🔍 Implementation Details

### Masked Convolution Mechanics

```python
# For a 5×5 kernel, center at (2,2)
# Mask Type A zeros out center and all future pixels
# Mask Type B allows center pixel

for i in range(kH):
    for j in range(kW):
        if i > yc or (i == yc and j > xc):
            mask[:, :, i, j] = 0
            
if mask_type == 'A':
    mask[:, :, yc, xc] = 0  # Zero out center
```

### Sequential Generation Process

1. Start with blank image (all zeros)
2. For each pixel position (top-left to bottom-right):
   - Pass current image through network
   - Get probability for current pixel
   - Sample pixel value from Bernoulli distribution
   - Update image with sampled value
3. Continue until all pixels generated

## 📚 Key Learning Outcomes

✅ **Autoregressive Modeling**: Understanding sequential probability modeling for images  
✅ **Masked Convolutions**: Implementing causal constraints in CNNs  
✅ **Residual Architectures**: Building deep networks with skip connections  
✅ **Generative Modeling**: Learning probability distributions over high-dimensional data  
✅ **Binary Image Generation**: Working with discrete pixel values  

## 🎓 Reflection Questions Answered

### Q1: How do generated images differ across training epochs?

**Answer**: At epoch 10, images are very blurry lacking structure. Epoch 20 shows reduced loss but unclear digits with emerging structure. Epoch 30 demonstrates clear improvement with well-structured, recognizable digits.

### Q2: Role of Masked Convolution

**Answer**: Masked convolution ensures autoregressive behavior by restricting each pixel to only see previous pixels during training and generation. This allows the CNN to predict one pixel at a time, enforcing sequential generation.

## 📖 References

- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) - van den Oord et al., 2016
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328) - van den Oord et al., 2016
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood](https://arxiv.org/abs/1701.05517)

## 🔧 Model Hyperparameters

- **Input channels**: 1 (grayscale)
- **Hidden channels**: 64
- **Residual blocks**: 4
- **Initial kernel size**: 7×7
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss function**: Binary Cross-Entropy with Logits
- **Training epochs**: 30 (recommended)
- **Batch size**: 64

## 🎨 Visualization

The project includes comprehensive visualization of:
- Original MNIST samples
- Noising process at different timesteps
- Generated samples at different training epochs
- Training/validation loss curves

## 🙏 Acknowledgments

Implementation based on the original PixelCNN papers by van den Oord et al., adapted for educational purposes on the MNIST dataset.
