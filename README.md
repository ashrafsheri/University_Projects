# Deep Learning Portfolio

A collection of deep learning projects showcasing implementations of modern neural network architectures and training techniques.

## 🚀 Projects

### 1. [Transformer LLM Implementation](./Transformer-LLM-Implementation)
**A from-scratch transformer language model following the LLaMA architecture**

- 🏗️ **Architecture**: 135M parameter model with grouped-query attention
- 🔑 **Key Features**: Rotary embeddings, RMSNorm, gated MLPs
- 📚 **Concepts**: Transformer blocks, attention mechanisms, autoregressive generation
- 🛠️ **Tech Stack**: PyTorch, transformers

**Highlights**: Complete implementation of modern transformer components including grouped-query attention for efficient inference and rotary position embeddings for better sequence modeling.

---

### 2. [PixelCNN Generative Model](./PixelCNN-Generative-Model)
**Autoregressive image generation using masked convolutions**

- 🎨 **Task**: Generate MNIST digits pixel-by-pixel
- 🔑 **Key Features**: Masked convolutions (Type A/B), residual blocks
- 📚 **Concepts**: Autoregressive modeling, causal constraints, sequential generation
- 🛠️ **Tech Stack**: PyTorch, torchvision

**Highlights**: Implements causal masking to ensure each pixel is generated conditionally on all previous pixels, demonstrating how CNNs can model sequential dependencies in images.

---

### 3. [DPO LLM Alignment](./DPO-LLM-Alignment)
**Direct Preference Optimization for aligning language models with human preferences**

- 🎯 **Task**: Align smolLM with human preferences without RL
- 🔑 **Key Features**: Contrastive preference learning, frozen reference model
- 📚 **Concepts**: RLHF without rewards, preference optimization, alignment
- 🛠️ **Tech Stack**: PyTorch, transformers, datasets

**Highlights**: Demonstrates modern alignment techniques that bypass complex reinforcement learning, using direct optimization on preference pairs for simpler and more stable training.

---

### 4. [Diffusion Models](./Diffusion-Models-MNIST)
**From DDPM to CLIP-guided text-to-image generation**

- 🌫️ **Tasks**: 
  - Unconditional DDPM
  - Class-conditional generation
  - CLIP-guided text-to-image
- 🔑 **Key Features**: Noise scheduling, U-Net with attention, CLIP guidance
- 📚 **Concepts**: Denoising diffusion, reverse processes, guidance
- 🛠️ **Tech Stack**: PyTorch, CLIP, transformers

**Highlights**: Progressive implementation from basic diffusion to text-guided generation, showcasing the power of iterative denoising and multi-modal conditioning.

---

### 5. [GNN Maze Solver](./GNN-Maze-Solver)
**Learning navigation policies using Graph Neural Networks**

- 🗺️ **Task**: Navigate mazes using learned GNN policies
- 🔑 **Key Features**: GraphSAGE, ego-graphs, imitation learning
- 📚 **Concepts**: Graph representation learning, spatial reasoning, policy networks
- 🛠️ **Tech Stack**: PyTorch Geometric, NetworkX

**Highlights**: Demonstrates how GNNs can learn effective navigation from local graph views, achieving near-optimal performance while using only partial information.

---

## 🎓 Technical Skills Demonstrated

### Deep Learning Fundamentals
- ✅ Neural network architecture design
- ✅ Loss function engineering
- ✅ Optimization strategies
- ✅ Regularization techniques
- ✅ Training dynamics analysis

### Advanced Architectures
- 🔷 **Transformers**: Self-attention, positional encodings, decoder stacks
- 🔷 **CNNs**: Masked convolutions, residual connections, U-Nets
- 🔷 **GNNs**: Message passing, neighborhood aggregation, graph pooling
- 🔷 **Diffusion Models**: Noise schedules, denoising networks, guidance

### Training Techniques
- 📊 Supervised learning from optimal demonstrations
- 📊 Preference-based optimization
- 📊 Multi-modal conditioning (text, class labels)
- 📊 Imitation learning from expert policies
- 📊 Efficient fine-tuning strategies

### Implementation Skills
- 💻 PyTorch ecosystem (torch, torchvision, torch-geometric)
- 💻 Model evaluation and visualization
- 💻 Data preprocessing pipelines
- 💻 Custom layer implementations
- 💻 Gradient computation and backpropagation

## 📊 Project Complexity Matrix

| Project | Architecture Complexity | Training Complexity | Novelty | Lines of Code |
|---------|------------------------|---------------------|---------|---------------|
| Transformer LLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~500 |
| PixelCNN | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~400 |
| DPO Alignment | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~300 |
| Diffusion Models | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~600 |
| GNN Maze Solver | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ~350 |

## 🛠️ Technology Stack

### Core Frameworks
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Transformers**: Pre-trained models and tokenizers
- **CLIP**: Vision-language models

### Supporting Libraries
- **NumPy**: Numerical computing
- **NetworkX**: Graph operations
- **Matplotlib**: Visualization
- **TQDM**: Progress bars
- **Datasets**: HuggingFace datasets

### Development Tools
- **Jupyter**: Interactive notebooks
- **Git**: Version control
- **Python 3.8+**: Programming language

## 📈 Learning Progression

The projects are ordered by conceptual complexity and build upon each other:

1. **Start**: Transformer basics (attention, embeddings)
2. **Progress**: Autoregressive models (PixelCNN)
3. **Advance**: Alignment techniques (DPO)
4. **Expand**: Generative models (Diffusion)
5. **Apply**: Graph learning (GNN)

## 🎯 Key Achievements

- ✅ Implemented 5 major deep learning architectures from scratch
- ✅ Trained models on diverse tasks (language, images, graphs)
- ✅ Applied modern techniques (attention, diffusion, preference learning)
- ✅ Achieved competitive performance vs. baselines
- ✅ Comprehensive documentation and visualization

## 📚 References & Inspiration

Each project includes detailed references to:
- Original research papers
- Implementation guides
- Theoretical background
- Best practices

## 🚀 Getting Started

Each project folder contains:
- 📄 **README.md**: Detailed project documentation
- 📓 **Notebooks**: Jupyter notebooks with implementations
- 🐍 **Scripts**: Python files for reusability
- 📊 **Results**: Training curves and generated samples

### Prerequisites

```bash
# Create virtual environment
python -m venv dl_env
source dl_env/bin/activate  # On Windows: dl_env\Scripts\activate

# Install dependencies (per project)
pip install torch torchvision
pip install transformers datasets
pip install torch-geometric
pip install networkx matplotlib tqdm
```

### Quick Navigation

```bash
# Clone and explore
cd Transformer-LLM-Implementation  # Start with transformers
cd ../PixelCNN-Generative-Model    # Explore autoregressive models
cd ../DPO-LLM-Alignment            # Learn alignment techniques
cd ../Diffusion-Models-MNIST       # Master diffusion models
cd ../GNN-Maze-Solver              # Apply graph learning
```

## 📝 Project Structure

Each project follows a consistent structure:

```
Project-Name/
├── README.md                 # Comprehensive documentation
├── notebook.ipynb           # Implementation notebook
├── script.py                # Python script version
└── (additional files)       # Model weights, utilities, etc.
```

## 🎓 Educational Value

These projects demonstrate:

1. **Theoretical Understanding**: Deep knowledge of architectures and algorithms
2. **Practical Implementation**: Ability to translate papers to code
3. **Debugging Skills**: Resolving training issues and convergence problems
4. **Optimization**: Efficient implementation and hyperparameter tuning
5. **Documentation**: Clear explanations and reproducible results

## 🌟 Highlights by Category

### 🏗️ Architecture Design
- Custom transformer blocks with modern components
- U-Net with attention for diffusion models
- GraphSAGE for graph representation learning
- Masked convolutions for causal modeling

### 🎯 Training Innovations
- Preference-based optimization (DPO)
- CLIP-guided generation
- Imitation learning from optimal policies
- Multi-task conditioning (class, text)

### 📊 Evaluation & Analysis
- Quantitative metrics (loss, accuracy, path length)
- Qualitative assessment (generated images, text)
- Comparative analysis (GNN vs. Dijkstra)
- Ablation studies (schedule, architecture choices)

## 🔮 Future Directions

Potential extensions:
- 🔄 Combine techniques (e.g., GNN + Diffusion for graph generation)
- 📈 Scale to larger models and datasets
- 🎮 Apply to real-world problems (robotics, NLP, computer vision)
- 🔬 Experiment with latest research (Mamba, Hyena, JEPA)

## 📬 Contact & Collaboration

These projects represent a comprehensive exploration of modern deep learning techniques, from foundational architectures to cutting-edge alignment methods.

---

**Portfolio Statistics**:
- 📁 5 Major Projects
- 📝 ~2,150 Lines of Code
- 🧠 10+ Neural Network Architectures
- 📚 50+ Research Papers Referenced
- ⏱️ 100+ Hours of Implementation

**Last Updated**: November 2025
