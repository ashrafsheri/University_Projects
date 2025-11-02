# Diffusion Models: From Theory to Text-Guided Generation

A comprehensive implementation of diffusion models including DDPM, class-conditional generation, and CLIP-guided text-to-image synthesis on MNIST.

## 🎯 Project Overview

This project explores three progressively advanced diffusion model techniques:

1. **Basic DDPM**: Unconditional image generation using denoising diffusion
2. **Class-Conditional Diffusion**: Generating specific digits (0-9) on demand
3. **CLIP-Guided Generation**: Text-to-image synthesis using natural language prompts

## 🧠 Theoretical Foundation

### Diffusion Process

Diffusion models learn to reverse a gradual noising process:

**Forward Process (Adding Noise)**:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Reverse Process (Denoising)**:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Key Properties

1. **Markov Chain**: Each step depends only on the previous step
2. **Closed-Form Forward**: Can jump to any timestep t directly:
   ```
   x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,  where ε ~ N(0,I)
   ```
3. **Learned Reverse**: Neural network predicts noise to remove

## 📁 Project Structure

```
Diffusion-Models-MNIST/
├── README.md                      # This file
├── 26100342-pa5-1.ipynb          # Part 1: DDPM Implementation
├── 26100342_PA5_2.ipynb          # Part 2: Class-Conditional Diffusion  
└── 26100342_PA5_2.py             # Python script version
```

## 🏗️ Architecture Components

### 1. Noise Schedules

Two scheduling strategies for controlling noise levels:

**Linear Schedule**:
```python
betas = torch.linspace(beta_start, beta_end, T)
# Simple, evenly distributed noise increase
```

**Cosine Schedule**:
```python
alpha_bar = cos((t/T + s)/(1+s) * π/2)^2
betas = 1 - alpha_bar[t] / alpha_bar[t-1]
# Smoother, often better for images
```

### 2. U-Net Architecture

Advanced U-Net with modern components:

```
Input (1×28×28)
    ↓
Conv 3×3 → 128 channels
    ↓
Encoder:
  Down Block 1: 128 → 128 (2× ResBlock + Downsample)
  Down Block 2: 128 → 256 (2× ResBlock + Downsample)
  Down Block 3: 256 → 512 (2× ResBlock + Downsample)
    ↓
Bottleneck:
  ResBlock → Attention → ResBlock
    ↓
Decoder:
  Up Block 1: 512 + 256 → 128 (Upsample + 2× ResBlock)
  Up Block 2: 128 + 128 → 128 (Upsample + 2× ResBlock)
    ↓
GroupNorm → SiLU → Conv 3×3 → 1 channel
```

**Key Components**:

- **Sinusoidal Time Embeddings**: Encode timestep information
- **Residual Blocks**: With time conditioning via FiLM (Feature-wise Linear Modulation)
- **Self-Attention**: At bottleneck for global context
- **Skip Connections**: U-Net style concatenation
- **Group Normalization**: Stable training

### 3. Time Conditioning

```python
class SinusoidalTimeEmbedding(nn.Module):
    def forward(self, t):
        # Convert timestep to sinusoidal features
        emb = log(10000) / (dim/2 - 1)
        emb = exp(arange(dim/2) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = cat([sin(emb), cos(emb)], dim=-1)
        return emb
```

## 🚀 Task Implementations

### Task 1: Basic DDPM

**Objective**: Generate unconditional MNIST digits

```python
# Forward diffusion: Add noise
x_t, noise = forward_diffusion_sample(x_0, t)

# Training: Predict the noise
noise_pred = model(x_t, t)
loss = F.mse_loss(noise_pred, noise)

# Sampling: Iteratively denoise
for t in reversed(range(T)):
    noise_pred = model(x_t, t)
    x_t = denoise_step(x_t, noise_pred, t)
```

**Key Insights**:
- Loss decreases from ~0.05 to ~0.01 over 15 epochs
- Early timesteps: blurry noise
- Late timesteps: recognizable digits emerge

### Task 2: Class-Conditional Generation

**Objective**: Generate specific digits on command

```python
class CondUNetMNIST(UNetMNIST):
    def __init__(self, n_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, time_dim)
    
    def forward(self, x, t, y):
        t_emb = self.time_embed(t)
        y_emb = self.label_emb(y)
        cond = t_emb + y_emb  # Combine embeddings
        # ... rest of forward pass with cond
```

**Training**:
```python
# Pass both timestep and class label
noise_pred = model(x_t, t, class_label)
loss = F.mse_loss(noise_pred, noise)
```

**Sampling**:
```python
# Generate digit '7'
target_label = 7
samples = sample_cond_ddpm(model, target_label, n_samples=1)
```

### Task 3: CLIP-Guided Generation

**Objective**: Generate images from text prompts like "two", "five", "nine"

**Architecture**:
```
Text Prompt → CLIP Text Encoder → Text Embedding (512-dim)
                                         ↓
                                   Cosine Similarity
                                         ↓
Generated Image → Resize → CLIP Image Encoder
```

**Guided Sampling**:
```python
def text_guided_ddpm(model, text_emb, guidance_scale=2.0):
    x = torch.randn(1, 1, 28, 28)  # Start from noise
    
    for t in reversed(range(T)):
        # 1. Predict noise and compute mean
        noise_pred = model(x, t)
        mu = compute_mean(x, noise_pred, t)
        
        # 2. CLIP guidance: push image toward text
        for _ in range(n_steps_per_t):
            img_for_clip = mu.clone().requires_grad_(True)
            clip_loss = clip_guidance_loss(img_for_clip, text_emb)
            clip_loss.backward()
            
            # Gradient descent on image
            x = x - lr * guidance_scale * img_for_clip.grad
        
        # 3. Add noise for next step (if not final)
        if t > 0:
            x = mu + sigma * torch.randn_like(x)
    
    return x
```

**CLIP Loss**:
```python
def clip_guidance_loss(img, text_emb):
    # Resize to 224×224, convert to 3 channels
    img = F.interpolate(img, size=(224, 224))
    img = img.repeat(1, 3, 1, 1)
    
    # Normalize for CLIP
    img = (img - CLIP_MEAN) / CLIP_STD
    
    # Get image embedding
    img_emb = clip_model.encode_image(img)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    # Minimize: 1 - cosine_similarity
    cos_sim = (img_emb * text_emb).sum(dim=-1)
    return 1.0 - cos_sim
```

## 📊 Results & Analysis

### DDPM (Task 1)

| Metric | Value |
|--------|-------|
| Training Epochs | 15 |
| Final MSE Loss | ~0.010 |
| Timesteps (T) | 1000 |
| Schedule | Cosine |
| Generation Time | ~30 seconds |

**Observations**:
- Cosine schedule produces smoother images than linear
- 1000 timesteps needed for quality; 20 steps too coarse

### Class-Conditional (Task 2)

Successfully generates all 10 digits (0-9) on demand with:
- Clear digit structure
- Correct class correspondence
- Minimal artifacts

### CLIP-Guided (Task 3)

| Prompt | Success Rate | Notes |
|--------|--------------|-------|
| "two" | ~80% | Sometimes generates 2-like shapes |
| "five" | ~70% | More challenging |
| "nine" | ~75% | Often successful |

**Sensitivity Analysis**:
- "two" vs "number two": Minimal difference (same semantic region)
- "dog": Produces fuzzy animal-like shapes (not digits)
- Guidance strength (β): Higher = stronger text alignment

## 🔍 Implementation Details

### Forward Diffusion

```python
def forward_diffusion_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    
    sqrt_alpha_hat = sqrt_alpha_hats[t]
    sqrt_one_minus_alpha_hat = sqrt_one_minus_ahats[t]
    
    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
    return x_t, noise
```

### Sampling (Reverse Diffusion)

```python
def sample_ddpm(model, shape=(1, 28, 28)):
    x = torch.randn(1, *shape)
    
    for t in reversed(range(T)):
        eps_pred = model(x, torch.tensor([t]))
        
        # Compute mean
        coef1 = 1 / sqrt(alphas[t])
        coef2 = betas[t] / sqrt(1 - alpha_hats[t])
        mu = coef1 * (x - coef2 * eps_pred)
        
        # Add noise (except last step)
        if t > 0:
            sigma = sqrt(posterior_variance[t])
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    
    return x.clamp(0, 1)
```

## 🎓 Learning Outcomes

✅ **Diffusion Theory**: Understanding forward and reverse processes  
✅ **Noise Scheduling**: Linear vs. cosine schedules and their effects  
✅ **U-Net Architecture**: Modern components (attention, residuals, time conditioning)  
✅ **Conditional Generation**: Class embedding and conditioning mechanisms  
✅ **CLIP Integration**: Text-image alignment for guided generation  
✅ **Training Dynamics**: Noise prediction, loss landscapes, convergence  

## 📝 Analytical Questions Answered

### Q1: Forward Process as Markov Chain

**Proof**: 
```
q(x_t | x_{0:t-1}) = q(x_t | x_{t-1})
```
Each step only depends on previous, not entire history. Assumption: Gaussian transitions with fixed variance schedule.

### Q2: Mean and Variance at Timestep t

**Derivation**:
```
x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
E[x_t | x_0] = √(ᾱ_t) x_0
Var[x_t | x_0] = (1-ᾱ_t) I
```

### Q3: Beta Schedule Effects

- **Too large early β**: Hard to reverse, poor generation
- **Too small β**: Requires many steps, slow sampling
- **Smooth schedules**: Better image quality, fewer artifacts

### Q4: Too Few Timesteps

With T=20: Large per-step β → coarse jumps → high discretization error → blurred, corrupted images.

## 📖 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, 2021
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - Ho & Salimans, 2022
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) - Radford et al., 2021

## 🔧 Hyperparameters

### DDPM (Task 1)
| Parameter | Value |
|-----------|-------|
| Timesteps (T) | 1000 |
| Schedule | Cosine |
| Learning Rate | 2e-4 |
| Epochs | 15 |
| Base Channels | 128 |
| Batch Size | 64 |

### Conditional (Task 2)
| Parameter | Value |
|-----------|-------|
| Classes | 10 |
| Base Channels | 64 |
| Time Embedding Dim | 128 |
| Epochs | 5 |

### CLIP-Guided (Task 3)
| Parameter | Value |
|-----------|-------|
| CLIP Model | ViT-B/32 |
| Guidance Scale | 2.0 |
| Steps per Timestep | 5 |
| CLIP Learning Rate | 0.15 |

## 🎨 Visualizations

The project includes:
- Forward diffusion progression (0 → T timesteps)
- Training loss curves
- Generated samples at different epochs
- Class-conditional digit grid (0-9)
- Text-guided generation results
- Path visualization for sampling process

## 🙏 Acknowledgments

- Implementation inspired by the DDPM paper and Hugging Face diffusers library
- CLIP model from OpenAI
- MNIST dataset from torchvision
- Educational adaptations for deep learning coursework

## 🌟 Key Takeaways

1. **Diffusion = Gradual Denoising**: Learn to reverse a noising process step-by-step
2. **Conditioning**: Add embeddings (class, text) to guide generation
3. **CLIP Power**: Enables zero-shot text-to-image without fine-tuning
4. **Schedule Matters**: Cosine > Linear for most image tasks
5. **Quality vs Speed**: More timesteps = better quality but slower sampling
