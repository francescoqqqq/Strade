#!/usr/bin/env python3
"""
Quick GPU check for nnU-Net training
"""

import sys

def check_gpu():
    """Check if GPU is available and properly configured"""
    print("\n" + "="*60)
    print("🔍 GPU CHECK")
    print("="*60 + "\n")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Install with: pip install torch torchvision")
        return False
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        
        # GPU info
        n_gpus = torch.cuda.device_count()
        print(f"✓ Number of GPUs: {n_gpus}")
        
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU tensor
        try:
            x = torch.rand(5, 3).cuda()
            print(f"✓ GPU tensor test: {x.device}")
            print("\n🎉 GPU is ready for training!")
            return True
        except Exception as e:
            print(f"✗ GPU tensor creation failed: {e}")
            return False
    else:
        print("\n⚠️  No GPU detected. Training will use CPU (SLOW!)")
        print("   For GPU support, install CUDA-enabled PyTorch:")
        print("   https://pytorch.org/get-started/locally/")
        return False

def check_nnunet():
    """Check nnU-Net installation"""
    print("\n" + "="*60)
    print("🔍 NNUNET CHECK")
    print("="*60 + "\n")
    
    try:
        import nnunetv2
        print(f"✓ nnU-Net v2 installed")
        
        # Check environment variables
        import os
        env_vars = {
            'nnUNet_raw': os.environ.get('nnUNet_raw'),
            'nnUNet_preprocessed': os.environ.get('nnUNet_preprocessed'),
            'nnUNet_results': os.environ.get('nnUNet_results')
        }
        
        all_set = True
        for var, value in env_vars.items():
            if value:
                print(f"✓ {var}: {value}")
            else:
                print(f"✗ {var}: NOT SET")
                all_set = False
        
        if not all_set:
            print("\n⚠️  Set environment variables:")
            print('   export nnUNet_raw="/workspace/nnUNet_raw"')
            print('   export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"')
            print('   export nnUNet_results="/workspace/nnUNet_results"')
        else:
            print("\n✓ All nnU-Net environment variables are set!")
        
        return all_set
        
    except ImportError:
        print("✗ nnU-Net v2 not installed!")
        print("  Install with: pip install nnunetv2")
        return False

def check_dataset():
    """Check if dataset is ready"""
    print("\n" + "="*60)
    print("🔍 DATASET CHECK")
    print("="*60 + "\n")
    
    import os
    import json
    
    dataset_dir = "/workspace/nnUNet_raw/Dataset001_Strade"
    preprocessed_dir = "/workspace/nnUNet_preprocessed/Dataset001_Strade"
    
    # Check raw dataset
    if os.path.exists(dataset_dir):
        print(f"✓ Raw dataset found: {dataset_dir}")
        
        # Check dataset.json
        dataset_json = os.path.join(dataset_dir, "dataset.json")
        if os.path.exists(dataset_json):
            with open(dataset_json) as f:
                data = json.load(f)
                n_training = data.get('numTraining', 0)
                print(f"  • Training images: {n_training}")
                print(f"  • Channels: {list(data.get('channel_names', {}).values())}")
                print(f"  • Labels: {list(data.get('labels', {}).keys())}")
        
        # Check images
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")
        
        if os.path.exists(images_dir):
            n_images = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
            print(f"  • Images found: {n_images}")
        
        if os.path.exists(labels_dir):
            n_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.png')])
            print(f"  • Labels found: {n_labels}")
    else:
        print(f"✗ Raw dataset not found: {dataset_dir}")
        return False
    
    # Check preprocessed dataset
    if os.path.exists(preprocessed_dir):
        print(f"✓ Preprocessed dataset found: {preprocessed_dir}")
        
        plans_file = os.path.join(preprocessed_dir, "nnUNetPlans.json")
        if os.path.exists(plans_file):
            with open(plans_file) as f:
                plans = json.load(f)
                print(f"  • Plans: {plans.get('plans_name')}")
                if '2d' in plans.get('configurations', {}):
                    config = plans['configurations']['2d']
                    print(f"  • Patch size: {config.get('patch_size')}")
                    print(f"  • Batch size: {config.get('batch_size')}")
    else:
        print(f"⚠️  Preprocessed dataset not found: {preprocessed_dir}")
        print("   Run: nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
        return False
    
    print("\n✓ Dataset is ready for training!")
    return True

if __name__ == "__main__":
    print("\n🚀 nnU-Net Setup Check for Dataset001_Strade\n")
    
    gpu_ok = check_gpu()
    nnunet_ok = check_nnunet()
    dataset_ok = check_dataset()
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"  GPU: {'✓ Ready' if gpu_ok else '✗ Not available (CPU training)'}")
    print(f"  nnU-Net: {'✓ Ready' if nnunet_ok else '✗ Not configured'}")
    print(f"  Dataset: {'✓ Ready' if dataset_ok else '✗ Not ready'}")
    print("="*60)
    
    if gpu_ok and nnunet_ok and dataset_ok:
        print("\n🎉 ALL CHECKS PASSED! Ready to train!")
        print("\nStart training with:")
        print("  nnUNetv2_train 1 2d 0  # Single fold")
        print("  nnUNetv2_train 1 2d all  # All folds (recommended)")
    else:
        print("\n⚠️  Some checks failed. Fix the issues above before training.")
        sys.exit(1)
    
    print()

