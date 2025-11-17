#!/usr/bin/env python3
"""
Script per testare le predizioni di nnU-Net su Dataset001_Strade
Mostra: Immagine Satellitare | Predizione | Ground Truth | Overlay
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ========== CONFIGURAZIONE ==========
dataset_name = "Dataset001_Strade"
dataset_id = "001"
fold = 0

# Directory
raw_data_dir = f"/workspace/nnUNet_raw/{dataset_name}"
images_dir = os.path.join(raw_data_dir, "imagesTr")
labels_dir = os.path.join(raw_data_dir, "labelsTr")

# Directory predizioni (usa validation set generato durante training)
predictions_dir = f"/workspace/nnUNet_results/{dataset_name}/nnUNetTrainer__nnUNetPlans__2d/fold_{fold}/validation"
output_dir = "/workspace/comparison_results"

# ========================================


def run_predictions_if_needed():
    """Esegue predizioni se non esistono già"""
    if not os.path.exists(predictions_dir) or len(os.listdir(predictions_dir)) == 0:
        print("\n🔮 Le predizioni non esistono ancora. Eseguile con:\n")
        print(f"nnUNetv2_predict -i {images_dir} -o {predictions_dir} -d {dataset_id} -c 2d -f {fold}\n")
        print("Questo comando genererà le predizioni per tutte le immagini di training.")
        print("Ci vorrà qualche minuto...\n")
        
        response = input("Vuoi eseguirlo ora? [y/n]: ").strip().lower()
        if response == 'y':
            import subprocess
            print("\n⏳ Esecuzione predizioni in corso...\n")
            cmd = f"nnUNetv2_predict -i {images_dir} -o {predictions_dir} -d {dataset_id} -c 2d -f {fold}"
            subprocess.run(cmd, shell=True)
            print("\n✓ Predizioni completate!\n")
        else:
            print("\n⚠️  Esegui prima le predizioni, poi rilancia questo script.")
            sys.exit(0)


def load_image_triple(image_name):
    """Carica immagine satellitare, predizione e ground truth"""
    # Nome base (senza _0000)
    base_name = image_name.replace('_0000.png', '')
    
    # Percorsi
    img_path = os.path.join(images_dir, image_name)
    pred_path = os.path.join(predictions_dir, base_name + '.png')
    gt_path = os.path.join(labels_dir, base_name + '.png')
    
    # Verifica esistenza
    if not os.path.exists(img_path):
        return None, None, None, f"Immagine non trovata: {img_path}"
    if not os.path.exists(pred_path):
        return None, None, None, f"Predizione non trovata: {pred_path}"
    if not os.path.exists(gt_path):
        return None, None, None, f"Ground truth non trovata: {gt_path}"
    
    # Carica
    img = np.array(Image.open(img_path))
    pred = np.array(Image.open(pred_path))
    gt = np.array(Image.open(gt_path))
    
    return img, pred, gt, None


def create_overlay(img, mask, color=[255, 0, 0], alpha=0.5):
    """Crea overlay colorato della maschera sull'immagine"""
    overlay = img.copy()
    mask_bool = mask > 0
    
    # Applica colore dove c'è la maschera
    for i in range(3):
        overlay[mask_bool, i] = (alpha * color[i] + (1 - alpha) * img[mask_bool, i]).astype(np.uint8)
    
    return overlay


def visualize_comparison(img, pred, gt, title="", save_path=None):
    """Visualizza confronto: Immagine | Predizione | Ground Truth | Overlay"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Immagine satellitare
    axes[0].imshow(img)
    axes[0].set_title('Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Predizione
    axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Ground Truth
    axes[2].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay (Rosso=GT, Verde=Pred, Giallo=Entrambi)
    overlay_combined = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    overlay_combined[gt > 0] = [255, 0, 0]      # Rosso: GT
    overlay_combined[pred > 0] = [0, 255, 0]    # Verde: Predizione
    overlap = (gt > 0) & (pred > 0)
    overlay_combined[overlap] = [255, 255, 0]   # Giallo: Match
    
    # Blend con immagine
    overlay_final = (0.6 * img + 0.4 * overlay_combined).astype(np.uint8)
    axes[3].imshow(overlay_final)
    axes[3].set_title('Overlay\n(Red=GT, Green=Pred, Yellow=Match)', fontsize=10, fontweight='bold')
    axes[3].axis('off')
    
    # Calcola metriche
    intersection = np.sum((pred > 0) & (gt > 0))
    union = np.sum((pred > 0) | (gt > 0))
    iou = intersection / union if union > 0 else 0
    
    dice = 2 * intersection / (np.sum(pred > 0) + np.sum(gt > 0)) if (np.sum(pred > 0) + np.sum(gt > 0)) > 0 else 0
    
    # Titolo con metriche
    fig.suptitle(f'{title}\nIoU: {iou:.4f} | Dice: {dice:.4f}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Salvato: {save_path}")
    
    return fig, (iou, dice)


def main():
    print("\n" + "="*70)
    print("🔍 VISUALIZZAZIONE PREDIZIONI vs GROUND TRUTH")
    print("="*70 + "\n")
    
    # Verifica predizioni
    run_predictions_if_needed()
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Lista immagini (solo quelle con predizioni nel validation set)
    pred_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.png')])
    all_images = [f.replace('.png', '_0000.png') for f in pred_files]
    
    if len(all_images) == 0:
        print("❌ Nessuna predizione trovata!")
        return
    
    print(f"📊 Trovate {len(all_images)} predizioni (validation set)\n")
    
    # Chiedi quante visualizzare
    print("Opzioni:")
    print("  1. Visualizza N campioni casuali")
    print("  2. Visualizza campioni specifici (es: 0, 10, 50)")
    print("  3. Visualizza tutti (crea solo file, non mostra)")
    
    choice = input("\nScelta [1/2/3]: ").strip()
    
    selected_images = []
    
    if choice == '1':
        num = int(input("Quanti campioni? [default: 5]: ").strip() or "5")
        selected_images = random.sample(all_images, min(num, len(all_images)))
    
    elif choice == '2':
        indices_str = input("Indici (separati da virgola, es: 0,10,50): ").strip()
        indices = [int(x.strip()) for x in indices_str.split(',')]
        selected_images = [all_images[i] for i in indices if i < len(all_images)]
    
    elif choice == '3':
        selected_images = all_images
        print(f"\n⚠️  Modalità batch: genererò {len(selected_images)} immagini senza visualizzarle.")
        show = False
    else:
        print("Scelta non valida. Uso default: 5 campioni casuali.")
        selected_images = random.sample(all_images, min(5, len(all_images)))
    
    show = choice != '3'
    
    print(f"\n🎨 Elaborazione {len(selected_images)} campioni...\n")
    
    # Metriche globali
    ious = []
    dices = []
    
    # Processa ogni immagine
    for idx, img_name in enumerate(selected_images, 1):
        base_name = img_name.replace('_0000.png', '')
        print(f"[{idx}/{len(selected_images)}] {base_name}")
        
        img, pred, gt, error = load_image_triple(img_name)
        
        if error:
            print(f"  ⚠️  {error}")
            continue
        
        # Crea visualizzazione
        save_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        fig, (iou, dice) = visualize_comparison(img, pred, gt, title=base_name, save_path=save_path)
        
        ious.append(iou)
        dices.append(dice)
        
        if show and idx <= 10:  # Mostra solo prime 10 per non intasare
            plt.show()
        else:
            plt.close(fig)
    
    # Statistiche finali
    print("\n" + "="*70)
    print("📊 STATISTICHE GLOBALI")
    print("="*70)
    print(f"Campioni elaborati: {len(ious)}")
    print(f"IoU medio:  {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Dice medio: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"IoU min:    {np.min(ious):.4f}")
    print(f"IoU max:    {np.max(ious):.4f}")
    print("="*70 + "\n")
    
    print(f"✓ Risultati salvati in: {output_dir}/")
    print()


if __name__ == "__main__":
    main()

