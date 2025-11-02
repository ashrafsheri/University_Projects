# Direct Preference Optimization (DPO) for LLM Alignment

Implementation of Direct Preference Optimization (DPO) to align language models with human preferences without reinforcement learning or reward modeling.

## 🎯 Project Overview

This project demonstrates how to align a pre-trained language model (smolLM-135M) with human preferences using DPO. Unlike traditional RLHF methods that require training a separate reward model and using PPO, DPO directly optimizes the policy based on preference data through a simple contrastive loss.

## 🧠 Key Concept: What is DPO?

**Direct Preference Optimization** simplifies the alignment process by:

1. **No Reward Model Needed**: Directly learns from preference pairs without training a separate reward model
2. **Simpler Training**: Uses standard supervised learning techniques instead of complex RL algorithms
3. **More Stable**: Avoids the instability often associated with PPO training

### The DPO Loss Function

```
L_DPO = -log(σ(β[(log π_θ(y_c|x) - log π_θ(y_r|x)) - (log π_ref(y_c|x) - log π_ref(y_r|x))]))
```

Where:
- `x`: The prompt
- `y_c`: The chosen (preferred) response
- `y_r`: The rejected response  
- `π_θ`: Current policy model (trainable)
- `π_ref`: Frozen reference model (baseline)
- `β`: Temperature hyperparameter (typically 0.1-0.5)
- `σ`: Sigmoid function

### Intuition

The model learns to increase the log-probability of chosen responses relative to rejected ones, beyond what the reference model already prefers. This creates a "preference margin" that guides the model toward human preferences.

## 📁 Project Structure

```
DPO-LLM-Alignment/
├── README.md                  # This file
└── 26100342_PA4.ipynb        # Complete DPO implementation notebook
```

## 🏗️ Implementation Components

### 1. Model Setup

```python
# Policy model (trainable)
policy_model = smolLM(config)
policy_model.load_state_dict(torch.load("full_finetuned_smolLM.pth"))

# Reference model (frozen)
reference_model = smolLM(config)
reference_model.load_state_dict(torch.load("full_finetuned_smolLM.pth"))
reference_model.eval()

# Freeze reference model
for param in reference_model.parameters():
    param.requires_grad = False
```

### 2. Dataset Processing

Uses the `Dahoas/full-hh-rlhf` dataset with strong preference filtering:

```python
def is_strong_preference(example):
    # Filter based on:
    # - Minimum length (>60 chars)
    # - Sufficient difference between chosen/rejected
    # - Low lexical overlap
    # - Appropriate stop word ratio
    # - More nouns/verbs in chosen response
    # - No weak phrases
    return True/False
```

### 3. Computing Log Probabilities

```python
def compute_logprob(model, input_ids, attention_mask):
    """Computes average log-probability per sample"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"][:, :-1, :]
    target_ids = input_ids[:, 1:]
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding and average
    masked = token_probs * attention_mask[:, 1:].float()
    return masked.sum(dim=1) / attention_mask[:, 1:].sum(dim=1).clamp(min=1)
```

### 4. DPO Loss Implementation

```python
def dpo_loss(policy_model, ref_model, chosen_inputs, rejected_inputs, beta=0.8):
    # Compute log-probs for chosen and rejected responses
    pi_chosen = compute_logprob(policy_model, **chosen_inputs)
    pi_rejected = compute_logprob(policy_model, **rejected_inputs)
    ref_chosen = compute_logprob(ref_model, **chosen_inputs)
    ref_rejected = compute_logprob(ref_model, **rejected_inputs)
    
    # Calculate preference margins
    policy_delta = pi_chosen - pi_rejected
    ref_delta = ref_chosen - ref_rejected
    
    # DPO loss
    logits_difference = beta * (policy_delta - ref_delta)
    loss = -F.logsigmoid(logits_difference).mean()
    
    return loss, logits_difference
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers datasets nltk
```

### Training Pipeline

```python
# 1. Load models
policy_model = smolLM(config).to(device)
reference_model = smolLM(config).to(device)

# 2. Load and filter dataset
dpo_dataset = load_dataset("Dahoas/full-hh-rlhf")
filtered_data = dpo_dataset["train"].filter(is_strong_preference)

# 3. Create data loaders
train_loader = DataLoader(DPODataset(train_data), batch_size=8, shuffle=True)

# 4. Training loop
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)

for batch in train_loader:
    loss, logits_diff = dpo_loss(policy_model, reference_model, 
                                  chosen_inputs, rejected_inputs)
    loss.backward()
    optimizer.step()
```

### Evaluation

```python
def compute_accuracy(logits_diff):
    """How often model prefers chosen over rejected"""
    return (logits_diff > 0).float().mean().item()

ref_acc = evaluate_dpo_accuracy(reference_model, test_loader)
policy_acc = evaluate_dpo_accuracy(policy_model, test_loader)

print(f"Reference Accuracy: {ref_acc * 100:.2f}%")
print(f"Policy Accuracy: {policy_acc * 100:.2f}%")
```

## 📊 Results & Observations

### Training Metrics

- **Dataset**: 1,000 strong preference pairs
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Beta (β)**: 0.8
- **Trainable Parameters**: ~25% of model (last 5 layers + LM head)

### Performance

The policy model should show:
- ✅ Decreasing DPO loss over training
- ✅ Increasing logits difference (stronger preference margin)
- ✅ Higher accuracy in preferring chosen responses
- ✅ Improved response quality compared to reference

## 🔍 Key Insights

### Preference Filtering

Strong preference filtering is crucial:
- Removes low-quality examples
- Ensures meaningful signal
- Improves training stability
- Leads to better alignment

### Beta (β) Parameter

Controls the tradeoff between:
- **High β**: Model stays close to reference (conservative)
- **Low β**: Model can diverge more (aggressive alignment)

Typical range: 0.1 to 0.5

### Frozen Layers

Freezing most of the model:
- Makes training feasible on limited hardware
- Preserves pre-trained knowledge
- Focuses alignment on final layers
- Reduces overfitting risk

## 🎓 Learning Outcomes

✅ **DPO Algorithm**: Understanding preference-based alignment without RL  
✅ **Contrastive Learning**: Using preference pairs for model training  
✅ **Log Probability Computation**: Calculating sequence likelihoods  
✅ **Model Comparison**: Reference vs. policy model evaluation  
✅ **Efficient Fine-tuning**: Selective parameter training strategies  

## 📝 Reflection Questions Answered

### Q1: Policy vs. Reference Accuracy

**Answer**: The policy model typically performs better due to fine-tuning on preference data. Factors limiting accuracy include:
1. Noisy or weak preference examples in the dataset
2. Poorly tuned hyperparameters (learning rate, β)

### Q2: Beta Parameter Role

**Answer**: β controls how strongly the policy stays close to the reference model. Too high → barely changes; too low → may overfit or diverge from meaningful outputs.

### Q3: Using Reference Log-Probs in Policy

**Answer**: This eliminates the preference signal, making the loss meaningless and preventing learning.

### Q4: Length Imbalance in Responses

**Answer**: Longer rejected responses can bias toward shorter outputs due to lower average log-probs. Solutions: normalize per token or filter such examples.

### Q5: Training Challenge

**Answer**: Common issue: both models accidentally computing from the same weights. Fix: ensure separate forward passes for policy and reference.

## 📖 References

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3) - GRPO variant

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | AdamW optimizer |
| Weight Decay | 1e-2 | L2 regularization |
| Beta (β) | 0.8 | DPO temperature |
| Batch Size | 8 | Training batch size |
| Dataset Size | 1,000 | Strong preference pairs |
| Frozen Layers | 25/30 | First 25 decoder layers |
| Max Sequence Length | 512 | Token limit |

## 🎯 Comparison: DPO vs PPO

| Aspect | DPO | PPO (Traditional RLHF) |
|--------|-----|------------------------|
| Reward Model | ❌ Not needed | ✅ Required |
| Training Complexity | 🟢 Simple | 🔴 Complex |
| Stability | 🟢 More stable | 🟡 Can be unstable |
| Computational Cost | 🟢 Lower | 🔴 Higher |
| Sample Efficiency | 🟢 Better | 🟡 Moderate |

## 🙏 Acknowledgments

- Based on the DPO paper by Rafailov et al.
- Uses SmolLM architecture by HuggingFace
- Dataset from Anthropic's HH-RLHF collection
- Built upon PA2 smolLM implementation

## 🌟 Future Directions

- Experiment with different β values
- Try other preference datasets
- Implement online DPO variants
- Compare with PPO/GRPO approaches
- Scale to larger models
