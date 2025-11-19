# 🚀 Quick Start Guide - Dataset001_Strade

**Quick reference for common commands and workflows**

---

## ✅ Pre-flight Checks

```bash
# Check if everything is ready (GPU, nnU-Net, Dataset)
python check_gpu.py

# Show dataset statistics
python visualize_samples.py --stats

# Visualize random samples
python visualize_samples.py --num 6 --output samples.png

# Visualize specific samples
python visualize_samples.py --indices 0 10 50 100 --output specific_samples.png
```

---

## 📦 Dataset Generation

### Generate new dataset

```bash
# Edit configuration in made_dataset.py:
# - osm_file: path to .osm.pbf file
# - num_images: number of images to generate
# - data: "imm,lab,all" (what to save)

python made_dataset.py
```

### Configuration options

| Variable | Description | Example |
|----------|-------------|---------|
| `osm_file` | OSM data file (.pbf) | `/workspace/belgium-roads.osm.pbf` |
| `dataset_id` | Dataset ID for nnU-Net | `"001"` |
| `dataset_name` | Dataset name | `"Strade"` |
| `num_images` | Number of images | `2000` |
| `image_size` | Image size in pixels | `512` |
| `data` | What to save | `"imm,lab,all"` |

**Data options:**
- `imm` → Satellite images (imagesTr/)
- `lab` → Binary road masks (labelsTr/)
- `all` → Satellite + roads overlay (allTr/)

---

## 🎓 Training Pipeline

### 1. Planning and Preprocessing

```bash
# Full preprocessing with dataset verification
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Only verification (if already preprocessed)
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity --clean
```

**Output:**
- `nnUNet_preprocessed/Dataset001_Strade/nnUNetPlans.json`
- `nnUNet_preprocessed/Dataset001_Strade/dataset_fingerprint.json`
- Preprocessed images in `nnUNetPlans_2d/`

### 2. Training

```bash
# Train single fold (faster, for testing)
nnUNetv2_train 1 2d 0

# Train all folds (recommended for final model)
nnUNetv2_train 1 2d all

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 0

# Resume training from checkpoint
nnUNetv2_train 1 2d 0 --continue_training
```

**Parameters:**
- `1` → Dataset ID (Dataset001)
- `2d` → Configuration (2D U-Net)
- `0` or `all` → Fold number

**Training time (RTX 3080):**
- Single fold: ~2-4 hours
- All 5 folds: ~10-20 hours

### 3. Find Best Configuration

```bash
# After training all folds, find best model
nnUNetv2_find_best_configuration 1 -c 2d
```

---

## 🔮 Inference

### Predict on new images

```bash
# Using single fold
nnUNetv2_predict \
  -i /path/to/input/images \
  -o /path/to/output/predictions \
  -d 1 \
  -c 2d \
  -f 0

# Using ensemble of all folds (better quality)
nnUNetv2_predict \
  -i /path/to/input/images \
  -o /path/to/output/predictions \
  -d 1 \
  -c 2d
```

**Input requirements:**
- Images must be RGB PNG files
- Name format: `*_0000.png` (e.g., `image001_0000.png`)
- Size: any (will be resized automatically)

**Output:**
- Binary masks (0/1) in PNG format
- Same name as input (without `_0000` suffix)

---

## 📊 Monitoring and Evaluation

### View training progress

```bash
# Training logs and checkpoints are in:
cd /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainer__nnUNetPlans__2d/fold_0/

# Key files:
# - training_log_*.txt: Training progress
# - checkpoint_best.pth: Best model
# - checkpoint_final.pth: Final model
# - progress.png: Loss curves (if available)
```

### Evaluate predictions

```bash
# Compare predictions with ground truth
nnUNetv2_evaluate_folder \
  /path/to/ground_truth \
  /path/to/predictions \
  -djfile /workspace/nnUNet_preprocessed/Dataset001_Strade/dataset.json
```

---

## 🛠️ Maintenance and Utilities

### Clean and restart

```bash
# Remove preprocessed data (to reprocess with different settings)
rm -rf /workspace/nnUNet_preprocessed/Dataset001_Strade/

# Remove training results (to retrain from scratch)
rm -rf /workspace/nnUNet_results/Dataset001_Strade/

# Remove raw dataset (to regenerate)
rm -rf /workspace/nnUNet_raw/Dataset001_Strade/
```

### Disk space management

```bash
# Check dataset sizes
du -sh /workspace/nnUNet_raw/Dataset001_Strade/*
du -sh /workspace/nnUNet_preprocessed/Dataset001_Strade/*
du -sh /workspace/nnUNet_results/Dataset001_Strade/*

# Remove visualization files (save space, not needed for training)
rm -rf /workspace/nnUNet_raw/Dataset001_Strade/labelsTr_viz/
rm -rf /workspace/nnUNet_raw/Dataset001_Strade/allTr/
```

### Export trained model

```bash
# Package model for sharing/deployment
nnUNetv2_export_model_to_zip \
  -d 1 \
  -o /workspace/model_export.zip \
  -c 2d \
  -f all
```

---

## 🔧 Troubleshooting

### GPU not detected

```bash
# Check GPU
python check_gpu.py

# If CUDA not available, install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory errors

```bash
# Reduce batch size by editing plans file:
# /workspace/nnUNet_preprocessed/Dataset001_Strade/nnUNetPlans.json
# Change "batch_size": 12 → 6 or 4

# Or train with smaller images:
# Edit made_dataset.py and set image_size = 256
```

### Training stuck or slow

```bash
# Check GPU usage
nvidia-smi

# Monitor training progress
tail -f /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_*.txt

# Check if using GPU (should see "cuda:0")
grep -i "cuda\|gpu" /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_*.txt
```

### Dataset issues

```bash
# Verify dataset integrity
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Check sample images
python visualize_samples.py --indices 0 1 2 3

# Verify label values (should be 0/1, not 0/255)
python -c '
from PIL import Image
import numpy as np
label = np.array(Image.open("/workspace/nnUNet_raw/Dataset001_Strade/labelsTr/strade_0000.png"))
print("Unique values:", np.unique(label))
'
```

---

## 📚 Additional Resources

### Documentation
- **README.md**: Complete project documentation
- **CHANGELOG.md**: Version history and features
- **requirements.txt**: Python dependencies

### Scripts
- **made_dataset.py**: Dataset generation
- **visualize_samples.py**: Dataset visualization
- **check_gpu.py**: System check

### nnU-Net Documentation
- GitHub: https://github.com/MIC-DKFZ/nnUNet
- Paper: https://arxiv.org/abs/1904.08128
- Docs: https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation

---

## 🎯 Typical Workflow

```bash
# 1. Check system
python check_gpu.py

# 2. Generate dataset (if not done)
python made_dataset.py

# 3. Visualize samples
python visualize_samples.py --num 10 --output samples.png

# 4. Preprocess
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 5. Train (test with single fold first)
nnUNetv2_train 1 2d 0

# 6. If results look good, train all folds
nnUNetv2_train 1 2d all

# 7. Find best configuration
nnUNetv2_find_best_configuration 1 -c 2d

# 8. Run inference
nnUNetv2_predict -i /path/to/images -o /path/to/output -d 1 -c 2d
```

---

**Quick Reference Card**

| Task | Command |
|------|---------|
| Check system | `python check_gpu.py` |
| Generate dataset | `python made_dataset.py` |
| View samples | `python visualize_samples.py --stats` |
| Preprocess | `nnUNetv2_plan_and_preprocess -d 1` |
| Train (test) | `nnUNetv2_train 1 2d 0` |
| Train (full) | `nnUNetv2_train 1 2d all` |
| Predict | `nnUNetv2_predict -i INPUT -o OUTPUT -d 1 -c 2d` |

---

**Last updated:** November 2025  
**Author:** Francesco Girardello

