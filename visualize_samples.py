#!/usr/bin/env python3
"""
Script per visualizzare campioni del dataset Dataset001_Strade
Mostra immagine satellitare, label binaria e overlay
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione
dataset_dir = "/workspace/nnUNet_raw/Dataset001_Strade"
images_dir = os.path.join(dataset_dir, "imagesTr")
labels_dir = os.path.join(dataset_dir, "labelsTr")
labels_viz_dir = os.path.join(dataset_dir, "labelsTr_viz")
num_samples = 4  # Numero di campioni da visualizzare

def load_sample(idx):
    """Carica immagine e label per un indice specifico"""
    img_name = f"strade_{idx:04d}_0000.png"
    lbl_name = f"strade_{idx:04d}.png"
    
    img_path = os.path.join(images_dir, img_name)
    lbl_path = os.path.join(labels_dir, lbl_name)
    lbl_viz_path = os.path.join(labels_viz_dir, lbl_name)
    
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        return None, None, None
    
    img = np.array(Image.open(img_path))
    label = np.array(Image.open(lbl_path))
    label_viz = np.array(Image.open(lbl_viz_path))
    
    return img, label, label_viz

def create_overlay(img, label):
    """Crea overlay rosso delle strade sull'immagine"""
    overlay = img.copy()
    # Evidenzia strade in rosso
    overlay[label == 1] = [255, 0, 0]
    # Blend con immagine originale
    result = (0.7 * img + 0.3 * overlay).astype(np.uint8)
    return result

def visualize_samples(indices=None, save_path=None):
    """Visualizza campioni del dataset"""
    if indices is None:
        # Seleziona campioni casuali
        all_files = sorted(os.listdir(images_dir))
        max_idx = len(all_files) - 1
        indices = random.sample(range(max_idx + 1), min(num_samples, max_idx + 1))
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        img, label, label_viz = load_sample(idx)
        
        if img is None:
            print(f"⚠️  Campione {idx} non trovato")
            continue
        
        overlay = create_overlay(img, label)
        
        # Calcola statistiche
        road_pixels = np.sum(label == 1)
        total_pixels = label.size
        road_percentage = (road_pixels / total_pixels) * 100
        
        # Visualizza
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Satellite Image\n(Sample {idx:04d})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(label_viz, cmap='gray')
        axes[i, 1].set_title(f'Road Mask\n({road_percentage:.2f}% roads)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay (Red)')
        axes[i, 2].axis('off')
        
        # Istogramma valori label
        axes[i, 3].bar(['Background\n(0)', 'Road\n(1)'], 
                       [np.sum(label == 0), np.sum(label == 1)],
                       color=['black', 'red'])
        axes[i, 3].set_title('Pixel Distribution')
        axes[i, 3].set_ylabel('Number of pixels')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Salvato in: {save_path}")
    else:
        plt.show()
    
    return fig

def print_dataset_stats():
    """Stampa statistiche del dataset"""
    print("\n" + "="*60)
    print("📊 STATISTICHE DATASET001_STRADE")
    print("="*60)
    
    # Conta file
    n_images = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
    n_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    print(f"\n📁 File:")
    print(f"  • Immagini: {n_images}")
    print(f"  • Label: {n_labels}")
    
    # Dimensione totale
    images_size = sum(os.path.getsize(os.path.join(images_dir, f)) 
                     for f in os.listdir(images_dir)) / (1024**2)
    labels_size = sum(os.path.getsize(os.path.join(labels_dir, f)) 
                     for f in os.listdir(labels_dir)) / (1024**2)
    
    print(f"\n💾 Dimensioni:")
    print(f"  • Immagini: {images_size:.1f} MB")
    print(f"  • Label: {labels_size:.1f} MB")
    print(f"  • Totale: {images_size + labels_size:.1f} MB")
    
    # Analizza alcuni campioni
    print(f"\n🔍 Analisi campioni (primi 100):")
    road_percentages = []
    for i in range(min(100, n_images)):
        _, label, _ = load_sample(i)
        if label is not None:
            road_pct = (np.sum(label == 1) / label.size) * 100
            road_percentages.append(road_pct)
    
    if road_percentages:
        print(f"  • Coverage strade medio: {np.mean(road_percentages):.2f}%")
        print(f"  • Coverage strade min: {np.min(road_percentages):.2f}%")
        print(f"  • Coverage strade max: {np.max(road_percentages):.2f}%")
        print(f"  • Coverage strade std: {np.std(road_percentages):.2f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizza campioni del dataset')
    parser.add_argument('--indices', type=int, nargs='+', 
                       help='Indici specifici da visualizzare (es: 0 10 50)')
    parser.add_argument('--num', type=int, default=4,
                       help='Numero di campioni casuali da visualizzare')
    parser.add_argument('--output', type=str,
                       help='Percorso file output (es: samples.png)')
    parser.add_argument('--stats', action='store_true',
                       help='Mostra solo statistiche dataset')
    
    args = parser.parse_args()
    
    if args.stats:
        print_dataset_stats()
    else:
        num_samples = args.num
        
        print(f"\n🖼️  Visualizzazione campioni dataset...\n")
        
        if args.indices:
            indices = args.indices
            print(f"Campioni selezionati: {indices}")
        else:
            # Seleziona casuali
            all_files = sorted(os.listdir(images_dir))
            max_idx = len(all_files) - 1
            indices = random.sample(range(max_idx + 1), min(num_samples, max_idx + 1))
            print(f"Campioni casuali: {indices}")
        
        fig = visualize_samples(indices, save_path=args.output)
        
        if not args.output:
            print("\n✓ Chiudi la finestra per uscire")

    print("\n✓ Fatto!\n")

