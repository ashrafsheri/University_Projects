# Transformer Language Model Implementation (smolLM)

A from-scratch implementation of a transformer-based language model following the LLaMA architecture, featuring grouped-query attention, rotary positional embeddings, and advanced architectural components.

## 🎯 Project Overview

This project implements a compact language model (smolLM) with 135M parameters, featuring:
- **Grouped-Query Attention (GQA)** for efficient multi-head attention
- **Rotary Position Embeddings (RoPE)** for better sequence modeling
- **RMS Normalization** for stable training
- **Gated MLP layers** with SiLU activation
- Complete decoder stack with residual connections

## 🏗️ Architecture

### Model Components

1. **Grouped-Query Attention**
   - 9 query heads, 3 key-value heads
   - Head dimension: 64
   - Implements efficient attention with reduced memory footprint

2. **Rotary Embeddings**
   - Base frequency: 10,000
   - Applied to queries and keys for position-aware attention

3. **Decoder Blocks**
   - 30 stacked decoder layers
   - Pre-normalization with RMSNorm
   - Gated FFN with intermediate size 1536

4. **Configuration**
   ```python
   vocab_size = 49,152
   hidden_size = 576
   intermediate_size = 1,536
   num_hidden_layers = 30
   num_heads = 9
   kv_heads = 3
   ```

## 📁 Project Structure

```
Transformer-LLM-Implementation/
├── README.md                 # This file
├── config.py                 # Model configuration dataclass
├── attention.py              # Grouped-query attention implementation
├── layers.py                 # Decoder blocks and MLP layers
├── model.py                  # Main smolLM model classes
├── generate.py               # Text generation utilities
├── test_model.py             # Model validation and testing
└── __pycache__/             # Python cache files
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch numpy transformers
```

### Quick Start

```python
from config import smolConfig
from model import smolLM
import torch

# Initialize model
config = smolConfig()
model = smolLM(config)

# Example forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 128))
attention_mask = torch.ones_like(input_ids)

outputs = model(input_ids, attention_mask)
logits = outputs['logits']  # Shape: (batch_size, seq_len, vocab_size)
```

### Text Generation

```python
from generate import generate_text
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
prompt = "The future of AI is"

generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=50
)
print(generated)
```

## 🔧 Implementation Details

### Grouped-Query Attention

GQA reduces the number of key-value heads while maintaining multiple query heads, providing a balance between:
- Multi-Query Attention (MQA): 1 KV head
- Multi-Head Attention (MHA): Equal query and KV heads

Benefits:
- Reduced memory usage during inference
- Faster decoding while maintaining model quality
- Better cache efficiency for auto-regressive generation

### Rotary Position Embeddings

RoPE encodes absolute positions with rotation matrices and naturally includes relative position information:

```python
# Frequency computation
freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

# Apply rotation to Q and K
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

### Weight Tying

The embedding layer and LM head share weights, reducing parameters and often improving performance:

```python
self.lm_head.weight = self.model.embed_tokens.weight
```

## 📊 Model Testing

Run the test suite to validate the implementation:

```bash
python test_model.py
```

Tests include:
- Forward pass correctness
- Output shape validation
- Gradient flow verification
- Attention mask application
- Generation functionality

## 🎓 Learning Outcomes

This implementation demonstrates:
- ✅ Modern transformer architecture patterns
- ✅ Efficient attention mechanisms (GQA)
- ✅ Advanced positional encoding (RoPE)
- ✅ Normalization techniques (RMSNorm)
- ✅ Residual connections and skip paths
- ✅ Auto-regressive language modeling

## 📚 References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [SmolLM by HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM-135M)

## 🔍 Key Features

- **Educational**: Clear, well-documented code for learning transformer internals
- **Modular**: Separated components for easy understanding and modification
- **Complete**: Full implementation from embeddings to generation
- **Tested**: Comprehensive test suite for validation

## 📝 Notes

- Model follows LLaMA architecture conventions
- Causal attention mask ensures auto-regressive generation
- RMS normalization provides training stability
- Gated MLP improves model expressiveness

## 🙏 Acknowledgments

Implementation based on the LLaMA and SmolLM architectures, with inspiration from the transformer literature and HuggingFace implementations.
