# Git Push Instructions

## Quick Start - Push All Projects to GitHub

### Step 1: Navigate to Projects Folder
```bash
cd /Users/ashrafshahreyar/Coding/Projects
```

### Step 2: Initialize Git (if not already done)
```bash
git init
```

### Step 3: Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Data files (optional - uncomment if you want to ignore data)
# *.pt
# *.pth
# *.h5
# data/
# *.csv
# *.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# Model weights (large files)
# *.bin
# *.safetensors
EOF
```

### Step 4: Add All Files
```bash
git add .
```

### Step 5: Commit
```bash
git commit -m "Initial commit: Deep Learning Portfolio with 5 major projects

- Transformer LLM Implementation (smolLM)
- PixelCNN Generative Model  
- DPO LLM Alignment
- Diffusion Models (DDPM, Conditional, CLIP-guided)
- GNN Maze Solver

Each project includes comprehensive README and implementations"
```

### Step 6: Create GitHub Repository
1. Go to https://github.com/new
2. Create a repository named: `deep-learning-portfolio` (or your preferred name)
3. **Don't** initialize with README, .gitignore, or license (we already have these)

### Step 7: Link and Push
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/deep-learning-portfolio.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Alternative: Individual Project Repositories

If you prefer separate repos for each project:

### Project 1: Transformer LLM
```bash
cd Transformer-LLM-Implementation
git init
git add .
git commit -m "Transformer language model implementation (smolLM)"
# Create repo on GitHub
git remote add origin https://github.com/YOUR_USERNAME/transformer-llm-implementation.git
git branch -M main
git push -u origin main
```

### Project 2: PixelCNN
```bash
cd ../PixelCNN-Generative-Model
git init
git add .
git commit -m "PixelCNN autoregressive image generation"
# Create repo on GitHub
git remote add origin https://github.com/YOUR_USERNAME/pixelcnn-generative-model.git
git branch -M main
git push -u origin main
```

### Project 3: DPO Alignment
```bash
cd ../DPO-LLM-Alignment
git init
git add .
git commit -m "Direct Preference Optimization for LLM alignment"
# Create repo on GitHub
git remote add origin https://github.com/YOUR_USERNAME/dpo-llm-alignment.git
git branch -M main
git push -u origin main
```

### Project 4: Diffusion Models
```bash
cd ../Diffusion-Models-MNIST
git init
git add .
git commit -m "Diffusion models: DDPM to CLIP-guided generation"
# Create repo on GitHub
git remote add origin https://github.com/YOUR_USERNAME/diffusion-models-mnist.git
git branch -M main
git push -u origin main
```

### Project 5: GNN Maze Solver
```bash
cd ../GNN-Maze-Solver
git init
git add .
git commit -m "Graph Neural Network maze navigation solver"
# Create repo on GitHub
git remote add origin https://github.com/YOUR_USERNAME/gnn-maze-solver.git
git branch -M main
git push -u origin main
```

---

## Recommendation

**Option 1 (Monorepo)**: Single repository `deep-learning-portfolio`
- ✅ Easier to manage
- ✅ Shows breadth of skills in one place
- ✅ Single README showcasing all projects
- ❌ Larger repository size

**Option 2 (Individual Repos)**: 5 separate repositories
- ✅ Focused project showcases
- ✅ Easier to share specific projects
- ✅ Can have different collaborators per project
- ❌ More repos to maintain

**My Suggestion**: Start with Option 1 (monorepo) for simplicity. You can always split later if needed.

---

## After Pushing

### Update Your GitHub Profile README
Add a "Projects" section:

```markdown
## 🚀 Featured Projects

### [Deep Learning Portfolio](https://github.com/YOUR_USERNAME/deep-learning-portfolio)
Collection of 5 advanced deep learning projects:
- 🤖 Transformer LLM (135M params, GQA, RoPE)
- 🎨 PixelCNN Generative Model
- 🎯 DPO LLM Alignment
- 🌫️ Diffusion Models (DDPM → CLIP-guided)
- 🗺️ GNN Maze Solver (GraphSAGE)

**Tech**: PyTorch, Transformers, PyTorch Geometric, CLIP
```

### Pin the Repository
1. Go to your GitHub profile
2. Click "Customize your pins"
3. Select your portfolio repository
4. It will appear at the top of your profile

---

## Troubleshooting

### Large Files Error
If you get errors about large files (>100MB):

```bash
# Find large files
find . -type f -size +50M

# Use Git LFS for large model weights
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Already Exists Error
If remote already exists:

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/repo-name.git
```

---

## Next Steps

1. ✅ Push to GitHub
2. 📝 Add project links to your resume/portfolio site
3. 🌟 Add badges to README (build status, license, etc.)
4. 📖 Write a blog post about one of the projects
5. 🔗 Share on LinkedIn/Twitter
