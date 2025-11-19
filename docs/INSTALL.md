# 🚀 Installation Guide - Dataset001_Strade

Quick setup guide for the road segmentation project.

---

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker (optional, for container setup)

---

## 🔧 Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/francescoqqqq/Strade.git
cd Strade
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- nnU-Net v2
- Geospatial libraries (geopandas, shapely, pyproj, fiona)
- Image processing (Pillow, opencv)
- Visualization (matplotlib)
- Scientific computing (numpy, scipy)
- HTTP client (requests)

### 3. Set up nnU-Net environment variables

```bash
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
```

**Permanent setup** (add to ~/.bashrc):

```bash
echo 'export nnUNet_raw="/workspace/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/workspace/nnUNet_results"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Fix torch.compile issue (if needed)

If you get CUDA linking errors during training:

```bash
# Create symlink for libcuda.so
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
ldconfig
```

### 5. Verify installation

```bash
python check_gpu.py
```

Expected output:
```
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 3080 (9.8 GB)
✓ nnU-Net v2 installed
✓ All nnU-Net environment variables are set!
✓ Dataset is ready for training!

🎉 ALL CHECKS PASSED! Ready to train!
```

---

## 🐳 Docker Setup (Alternative)

If using the provided devcontainer:

```bash
# In VS Code: Reopen in Container
# Or manually:
docker build -t strade-nnunet .devcontainer/
docker run -it --gpus all strade-nnunet
```

Inside the container:

```bash
pip install -r requirements.txt
```

---

## 🧪 Quick Test

Test dataset generation with 10 samples:

```bash
# Edit made_dataset.py: set num_images = 10
python made_dataset.py
```

---

## ⚠️ Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'geopandas'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 2: `torch.compile` fails with CUDA linking error

**Solution:**
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
ldconfig
```

### Issue 3: Git push fails with Git LFS error

**Solution:**
```bash
rm -f .git/hooks/pre-push .git/hooks/post-checkout .git/hooks/post-commit
git push origin main
```

### Issue 4: "dubious ownership" error in Docker

**Solution:**
```bash
git config --global --add safe.directory /workspace
```

---

## 📚 Next Steps

After successful installation:

1. **Generate dataset:** `python made_dataset.py`
2. **Preprocess:** `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity`
3. **Train:** `nnUNetv2_train 1 2d 0`
4. **Evaluate:** `python test_predictions.py`

See [QUICK_START.md](QUICK_START.md) for detailed workflows.

---

**Installation complete!** 🎉

