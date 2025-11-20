#!/usr/bin/env python3
"""
Script per analizzare i campioni problematici del dataset
Identifica immagini con problemi di ground truth o occlusioni
"""

import os
import json
import numpy as np
from PIL import Image  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from pathlib import Path

# Configurazione
dataset_name = "Dataset001_Strade"
raw_data_dir = f"/workspace/nnUNet_raw/{dataset_name}"
images_dir = os.path.join(raw_data_dir, "imagesTr")
labels_dir = os.path.join(raw_data_dir, "labelsTr")
output_dir = "/workspace/risultati/analisi_dataset"

os.makedirs(output_dir, exist_ok=True)

def analyze_sample(img_path, label_path):
    """Analizza un campione per identificare potenziali problemi"""
    img = np.array(Image.open(img_path))
    label = np.array(Image.open(label_path))
    
    # Statistiche
    road_pixels = np.sum(label > 0)
    total_pixels = label.size
    road_percentage = road_pixels / total_pixels * 100
    
    # Analisi colore immagine (per rilevare vegetazione/ombre)
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))
    
    # Indice di vegetazione semplificato (NDVI-like)
    # Verde alto + Rosso basso = vegetazione
    vegetation_index = mean_rgb[1] - mean_rgb[0]  # G - R
    
    # Brightness (per rilevare ombre)
    brightness = mean_rgb.mean()
    
    # Texture variance (immagini con alta variance = molti dettagli)
    texture_variance = img.std()
    
    return {
        'road_percentage': road_percentage,
        'mean_rgb': mean_rgb.tolist(),
        'std_rgb': std_rgb.tolist(),
        'vegetation_index': float(vegetation_index),
        'brightness': float(brightness),
        'texture_variance': float(texture_variance),
        'road_pixels': int(road_pixels)
    }


def identify_problematic_samples():
    """Identifica campioni potenzialmente problematici"""
    print("\n" + "="*70)
    print("🔍 ANALISI CAMPIONI PROBLEMATICI")
    print("="*70 + "\n")
    
    # Lista tutti i campioni
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    print(f"📊 Analisi di {len(label_files)} campioni...")
    
    results = []
    
    for i, label_file in enumerate(label_files, 1):
        if i % 200 == 0:
            print(f"  Processati {i}/{len(label_files)}...")
        
        base_name = label_file.replace('.png', '')
        img_file = base_name + '_0000.png'
        
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(img_path):
            continue
        
        stats = analyze_sample(img_path, label_path)
        stats['filename'] = base_name
        results.append(stats)
    
    print(f"✓ Analisi completata!\n")
    
    # Converti in array per analisi
    results_array = np.array([(
        r['road_percentage'],
        r['vegetation_index'],
        r['brightness'],
        r['texture_variance']
    ) for r in results])
    
    # Identifica categorie problematiche
    print("="*70)
    print("📋 CATEGORIE PROBLEMATICHE")
    print("="*70 + "\n")
    
    # 1. Campioni con TROPPA vegetazione (probabile occlusione)
    vegetation_threshold = np.percentile([r['vegetation_index'] for r in results], 75)
    high_vegetation = [r for r in results if r['vegetation_index'] > vegetation_threshold and r['road_percentage'] > 5]
    
    print(f"🌳 Alta vegetazione (possibili occlusioni): {len(high_vegetation)} campioni")
    print(f"   Threshold: vegetation_index > {vegetation_threshold:.2f}")
    if len(high_vegetation) > 0:
        print(f"   Top 5 peggiori:")
        sorted_veg = sorted(high_vegetation, key=lambda x: x['vegetation_index'], reverse=True)[:5]
        for r in sorted_veg:
            print(f"     - {r['filename']}: veg_idx={r['vegetation_index']:.2f}, strada={r['road_percentage']:.1f}%")
    
    # 2. Campioni con POCHE strade (possibile GT incompleto o zona errata)
    low_road = [r for r in results if r['road_percentage'] < 2]
    print(f"\n🛣️  Poche strade (< 2% pixel): {len(low_road)} campioni")
    if len(low_road) > 0:
        print(f"   Top 5 con meno strade:")
        sorted_road = sorted(low_road, key=lambda x: x['road_percentage'])[:5]
        for r in sorted_road:
            print(f"     - {r['filename']}: {r['road_percentage']:.2f}% strade")
    
    # 3. Campioni molto scuri (ombre, sera, nuvole)
    dark_threshold = np.percentile([r['brightness'] for r in results], 25)
    dark_samples = [r for r in results if r['brightness'] < dark_threshold]
    print(f"\n🌑 Immagini scure (possibili ombre/nuvole): {len(dark_samples)} campioni")
    print(f"   Threshold: brightness < {dark_threshold:.2f}")
    
    # 4. Campioni con MOLTE strade (potenziale area urbana complessa)
    high_road = [r for r in results if r['road_percentage'] > 15]
    print(f"\n🏙️  Molte strade (> 15% pixel, aree urbane dense): {len(high_road)} campioni")
    if len(high_road) > 0:
        print(f"   Top 5 con più strade:")
        sorted_high = sorted(high_road, key=lambda x: x['road_percentage'], reverse=True)[:5]
        for r in sorted_high:
            print(f"     - {r['filename']}: {r['road_percentage']:.1f}% strade")
    
    # Salva statistiche complete
    stats_file = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Statistiche complete salvate in: {stats_file}")
    
    # Crea visualizzazione distribuzione
    create_distribution_plots(results)
    
    # Genera lista campioni da rivedere manualmente
    samples_to_check = set()
    samples_to_check.update([r['filename'] for r in high_vegetation[:20]])  # Top 20 vegetazione
    samples_to_check.update([r['filename'] for r in low_road[:20]])  # Top 20 poche strade
    samples_to_check.update([r['filename'] for r in dark_samples[:20]])  # Top 20 scure
    
    check_file = os.path.join(output_dir, "samples_to_review.txt")
    with open(check_file, 'w') as f:
        f.write("# Campioni da rivedere manualmente\n")
        f.write(f"# Totale: {len(samples_to_check)} campioni\n\n")
        for sample in sorted(samples_to_check):
            f.write(f"{sample}\n")
    
    print(f"📝 Lista {len(samples_to_check)} campioni da rivedere: {check_file}")
    
    return results


def create_distribution_plots(results):
    """Crea grafici di distribuzione delle statistiche"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribuzione Statistiche Dataset', fontsize=16, fontweight='bold')
    
    # 1. Percentuale strade
    axes[0, 0].hist([r['road_percentage'] for r in results], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Percentuale strade (%)')
    axes[0, 0].set_ylabel('Numero campioni')
    axes[0, 0].set_title('Distribuzione Strade')
    axes[0, 0].axvline(np.median([r['road_percentage'] for r in results]), color='red', linestyle='--', label='Mediana')
    axes[0, 0].legend()
    
    # 2. Indice vegetazione
    axes[0, 1].hist([r['vegetation_index'] for r in results], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Indice Vegetazione (G-R)')
    axes[0, 1].set_ylabel('Numero campioni')
    axes[0, 1].set_title('Distribuzione Vegetazione')
    axes[0, 1].axvline(np.median([r['vegetation_index'] for r in results]), color='red', linestyle='--', label='Mediana')
    axes[0, 1].legend()
    
    # 3. Brightness
    axes[0, 2].hist([r['brightness'] for r in results], bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Brightness (media RGB)')
    axes[0, 2].set_ylabel('Numero campioni')
    axes[0, 2].set_title('Distribuzione Luminosità')
    axes[0, 2].axvline(np.median([r['brightness'] for r in results]), color='red', linestyle='--', label='Mediana')
    axes[0, 2].legend()
    
    # 4. Texture variance
    axes[1, 0].hist([r['texture_variance'] for r in results], bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Varianza Texture')
    axes[1, 0].set_ylabel('Numero campioni')
    axes[1, 0].set_title('Distribuzione Texture')
    axes[1, 0].axvline(np.median([r['texture_variance'] for r in results]), color='red', linestyle='--', label='Mediana')
    axes[1, 0].legend()
    
    # 5. Scatter: Vegetazione vs Strade
    axes[1, 1].scatter([r['vegetation_index'] for r in results], 
                       [r['road_percentage'] for r in results], 
                       alpha=0.5, s=10, color='green')
    axes[1, 1].set_xlabel('Indice Vegetazione')
    axes[1, 1].set_ylabel('Percentuale Strade (%)')
    axes[1, 1].set_title('Vegetazione vs Strade')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scatter: Brightness vs Strade
    axes[1, 2].scatter([r['brightness'] for r in results], 
                       [r['road_percentage'] for r in results], 
                       alpha=0.5, s=10, color='orange')
    axes[1, 2].set_xlabel('Brightness')
    axes[1, 2].set_ylabel('Percentuale Strade (%)')
    axes[1, 2].set_title('Luminosità vs Strade')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "dataset_distributions.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Grafici distribuzione salvati in: {plot_file}")
    plt.close()


if __name__ == "__main__":
    results = identify_problematic_samples()
    
    print("\n" + "="*70)
    print("✅ ANALISI COMPLETATA")
    print("="*70)
    print(f"\n📁 Risultati salvati in: {output_dir}/")
    print("   - dataset_statistics.json (statistiche complete)")
    print("   - dataset_distributions.png (grafici)")
    print("   - samples_to_review.txt (campioni da controllare)")
    print()

