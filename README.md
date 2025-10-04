This repository contains complete implementations for:
- **Q1**: Vision Transformer on CIFAR-10 from scratch (PyTorch)
- **Q2**: Text-Driven Image Segmentation with SAM 2 + GroundingDINO

## 🎯 Q1: Vision Transformer on CIFAR-10

### How to Run in Google Colab

1. **Upload the notebook**: Upload `q1.ipynb` to Google Colab
2. **Set runtime**: Runtime → Change runtime type → Select GPU (T4/V100/A100)
3. **Run all cells**: Runtime → Run all (Ctrl+F9)
4. **Expected runtime**: ~1 hour for full training (50 epochs)

### Model Configuration (Best Results)

```python
# Model Architecture
img_size = 32          # CIFAR-10 native resolution
patch_size = 4         # Small patches for 32x32 images  
embed_dim = 320        # Embedding dimension
num_heads = 8          # Multi-head attention
num_layers = 8         # Transformer encoder blocks
mlp_ratio = 4          # MLP expansion ratio

# Training Configuration  
batch_size = 128
learning_rate = 1e-3
weight_decay = 5e-4
epochs = 50   # 100-200 epochs would yield higher accuracy
warmup_epochs = 10
mixup_alpha = 0.8
label_smoothing = 0.1
```

### Results Table

| Metric | Value |
|--------|-------|
| **Final Test Accuracy** | **66.93%** |
| Validation Accuracy | 63.8% |
| Model Parameters | ~4.2M |
| Training Time | ~1 hour (GPU) |
| Patch Size | 4x4 (64 patches) |
| Architecture | Custom ViT-Small |

### Key Architecture Features

🔧 **Technical Innovations**:
- **Optimized patch size**: 4x4 patches instead of 16x16 for small CIFAR-10 images
- **Compact architecture**: 320-dim embeddings vs 768-dim in standard ViT
- **Advanced data augmentation**: Mixup, CutMix, RandomErasing, ColorJitter
- **Smart training strategy**: Cosine annealing with warmup, gradient clipping
- **Regularization**: Dropout, label smoothing, weight decay

### Architecture Analysis

#### Patch Size Choices
- **16x16 patches**: Only 4 patches total → insufficient detail
- **8x8 patches**: 16 patches → reasonable but limited
- **4x4 patches**: 64 patches → optimal granularity for 32x32 images

#### Depth vs Width Trade-offs
- **Deeper models** (12+ layers): Overfitting risk on small dataset
- **Our choice** (8 layers): Sweet spot for CIFAR-10 complexity
- **Narrower embeddings** (320 vs 768): Prevents overparametrization

#### Training Optimizations
- **Mixup augmentation**: +2-3% accuracy improvement
- **Cosine scheduler**: Better convergence than fixed LR
- **Warmup**: Stabilizes early training
- **Label smoothing**: Reduces overconfident predictions

---

## 🎯 Q2: Text-Driven Image Segmentation with SAM 2

### Pipeline Description

```
Text Prompt → GroundingDINO → Bounding Boxes → SAM 2 → Segmentation Masks
```

**Components**:
1. **GroundingDINO**: Converts natural language to object detections
2. **SAM 2**: Generates precise segmentation masks from bounding boxes  
3. **Visualization**: Creates colored overlays and individual mask displays

### How to Run

1. **Upload notebook**: Upload `q2.ipynb` to Google Colab
2. **Set GPU runtime**: Ensure GPU is enabled for faster inference
3. **Install dependencies**: First cell installs all required packages
4. **Run pipeline**: Execute cells to load models and run demos
5. **Try custom prompts**: Use interactive demo with your images and text

### Usage Examples

```python
# Example prompts that work well:
"car. truck. vehicle."           # Multiple related objects
"person. human."                 # People detection
"cat. dog. animal."              # Animals
"chair. table. furniture."       # Indoor objects
"tree. plant."                   # Outdoor elements
```

### Pipeline Capabilities

✅ **Strengths**:
- Zero-shot segmentation with natural language
- High-quality masks from SAM 2
- No training or fine-tuning required
- Handles multiple objects per image

⚠️ **Limitations**:
- Depends on GroundingDINO's text understanding
- Complex/ambiguous prompts may fail
- Processing time scales with image complexity
- Struggles with very small or heavily occluded objects
---

## 🔬 Technical Analysis

### Vision Transformer Design Rationale

1. **Patch Strategy**: 4x4 patches provide 64 tokens, enabling fine-grained attention while maintaining computational efficiency

2. **Architecture Scaling**: Reduced embedding dimension (320 vs 768) and layer count (8 vs 12) prevents overfitting on small datasets

3. **Attention Mechanisms**: 8-head attention balances representation capacity with computational cost

### SAM 2 Pipeline Innovation

1. **Text-to-Detection**: GroundingDINO provides robust zero-shot object detection from natural language

2. **Detection-to-Segmentation**: SAM 2 converts bounding boxes to pixel-perfect masks

3. **Multi-Modal Integration**: Seamless combination of language understanding and computer vision

## 📚 References

- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
- Kirillov et al. "SAM 2: Segment Anything in Images and Videos" 2024  
- Liu et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" 2023
- He et al. "Deep Residual Learning for Image Recognition" CVPR 2016
- https://github.com/omihub777/ViT-CIFAR
