#!/usr/bin/env python3
"""
Script per testare le predizioni di nnU-Net su Dataset001_Strade
Mostra: Immagine Satellitare | Predizione | Ground Truth | Overlay
Salva risultati in: risultati/risultati_test.txt e risultati/Confronto_imm/
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random
from datetime import datetime

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

# Directory output organizzata
output_base_dir = "/workspace/risultati"
output_images_dir = os.path.join(output_base_dir, "Confronto_imm")
output_report_file = os.path.join(output_base_dir, "risultati_test.txt")

# ========== IMPOSTAZIONI TEST ==========
# MODE: 'all', 'random', 'specific'
TEST_MODE = 'random'           # 'all' = tutte, 'random' = N casuali, 'specific' = numeri specifici

# Se TEST_MODE = 'random':
NUM_RANDOM = 20             # Numero di immagini casuali da testare

# Se TEST_MODE = 'specific':
SPECIFIC_IMAGES = [166, 317, 351, 485, 711, 930, 1186, 1332, 1496, 1797]  # Numeri delle immagini (strade_XXXX)

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


def calculate_metrics(pred, gt):
    """Calcola Dice, IoU e Accuracy"""
    pred_bool = pred > 0
    gt_bool = gt > 0
    
    # Dice Coefficient
    intersection = np.sum(pred_bool & gt_bool)
    dice = 2.0 * intersection / (np.sum(pred_bool) + np.sum(gt_bool)) if (np.sum(pred_bool) + np.sum(gt_bool)) > 0 else 0.0
    
    # IoU
    union = np.sum(pred_bool | gt_bool)
    iou = intersection / union if union > 0 else 0.0
    
    # Pixel Accuracy
    accuracy = np.sum(pred_bool == gt_bool) / pred_bool.size
    
    return dice, iou, accuracy


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
    dice, iou, accuracy = calculate_metrics(pred, gt)
    
    # Titolo con metriche
    fig.suptitle(f'{title}\nDice: {dice:.4f} | IoU: {iou:.4f} | Accuracy: {accuracy:.4f}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Salvato: {os.path.basename(save_path)}")
    
    plt.close(fig)
    
    return dice, iou, accuracy


def write_report(results, report_file):
    """Scrive report con tabelle metriche"""
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("═" * 80 + "\n")
        f.write("  REPORT RISULTATI TEST - SEGMENTAZIONE STRADE\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Numero campioni: {len(results)}\n")
        f.write(f"Checkpoint: checkpoint_best.pth\n")
        f.write("\n")
        
        # Calcola metriche una volta
        dices = [r[1] for r in results]
        ious = [r[2] for r in results]
        accs = [r[3] for r in results]
        
        # ═══ PRIMA: STATISTICHE GLOBALI ═══
        f.write("═" * 80 + "\n")
        f.write("STATISTICHE GLOBALI\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"{'Metrica':<20} {'Media':>12} {'Std Dev':>12} {'Min':>10} {'Max':>10}\n")
        f.write("─" * 80 + "\n")
        
        f.write(f"{'Dice Coefficient':<20} "
                f"{np.mean(dices):>12.4f} "
                f"{np.std(dices):>12.4f} "
                f"{np.min(dices):>10.4f} "
                f"{np.max(dices):>10.4f}\n")
        
        f.write(f"{'IoU':<20} "
                f"{np.mean(ious):>12.4f} "
                f"{np.std(ious):>12.4f} "
                f"{np.min(ious):>10.4f} "
                f"{np.max(ious):>10.4f}\n")
        
        f.write(f"{'Pixel Accuracy':<20} "
                f"{np.mean(accs):>12.4f} "
                f"{np.std(accs):>12.4f} "
                f"{np.min(accs):>10.4f} "
                f"{np.max(accs):>10.4f}\n")
        
        f.write("\n")
        
        # ═══ POI: TABELLA RISULTATI INDIVIDUALI ═══
        f.write("═" * 80 + "\n")
        f.write("METRICHE PER OGNI IMMAGINE\n")
        f.write("═" * 80 + "\n\n")
        
        f.write(f"{'Immagine':<25} {'Dice':>10} {'IoU':>10} {'Accuracy':>10}\n")
        f.write("─" * 80 + "\n")
        
        for img_name, dice, iou, acc in results:
            base_name = img_name.replace('_0000.png', '')
            f.write(f"{base_name:<25} {dice:>10.4f} {iou:>10.4f} {acc:>10.4f}\n")
        
        f.write("─" * 80 + "\n\n")
        
        # Performance breakdown
        f.write("─" * 80 + "\n")
        f.write("DISTRIBUZIONE PERFORMANCE\n")
        f.write("─" * 80 + "\n\n")
        
        excellent = sum(1 for d in dices if d > 0.95)
        good = sum(1 for d in dices if 0.85 <= d <= 0.95)
        fair = sum(1 for d in dices if 0.70 <= d < 0.85)
        poor = sum(1 for d in dices if d < 0.70)
        
        total = len(dices)
        
        f.write(f"Eccellenti (Dice > 0.95):     {excellent:3d}/{total} ({excellent/total*100:5.1f}%)\n")
        f.write(f"Buone (Dice 0.85-0.95):       {good:3d}/{total} ({good/total*100:5.1f}%)\n")
        f.write(f"Discrete (Dice 0.70-0.85):    {fair:3d}/{total} ({fair/total*100:5.1f}%)\n")
        f.write(f"Problematiche (Dice < 0.70):  {poor:3d}/{total} ({poor/total*100:5.1f}%)\n")
        
        f.write("\n" + "═" * 80 + "\n")
        f.write("File immagini salvati in: risultati/Confronto_imm/\n")
        f.write("═" * 80 + "\n")
    
    print(f"\n✅ Report salvato in: {report_file}")


def main():
    print("\n" + "="*70)
    print("🔍 VISUALIZZAZIONE PREDIZIONI vs GROUND TRUTH")
    print("="*70 + "\n")
    
    # Verifica predizioni
    run_predictions_if_needed()
    
    # Crea directory output
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    print(f"📁 Output directory: {output_base_dir}/")
    print(f"   - Immagini: {output_images_dir}/")
    print(f"   - Report: {output_report_file}\n")
    
    # Lista immagini (solo quelle con predizioni nel validation set)
    pred_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('.png')])
    all_images = [f.replace('.png', '_0000.png') for f in pred_files]
    
    if len(all_images) == 0:
        print("❌ Nessuna predizione trovata!")
        return
    
    print(f"📊 Trovate {len(all_images)} predizioni (validation set)")
    
    # Seleziona immagini in base alla configurazione
    selected_images = []
    
    if TEST_MODE == 'random':
        num = min(NUM_RANDOM, len(all_images))
        selected_images = random.sample(all_images, num)
        print(f"📋 Modalità: RANDOM - {num} immagini casuali")
    
    elif TEST_MODE == 'specific':
        # Cerca le immagini per NOME invece che per indice dell'array
        requested_names = [f'strade_{str(i).zfill(4)}_0000.png' for i in SPECIFIC_IMAGES]
        selected_images = [img for img in all_images if img in requested_names]
        
        if len(selected_images) < len(requested_names):
            missing = set(requested_names) - set(selected_images)
            print(f"⚠️  Alcune immagini non trovate nel validation set: {[m.replace('_0000.png', '') for m in missing]}")
        
        print(f"📋 Modalità: SPECIFIC - {len(selected_images)}/{len(requested_names)} immagini trovate")
    
    else:  # 'all' o default
        selected_images = all_images
        print(f"📋 Modalità: ALL - Tutte le {len(all_images)} immagini")
    
    print(f"\n🎨 Elaborazione {len(selected_images)} campioni...\n")
    
    # Lista per raccogliere risultati
    results = []
    
    # Processa ogni immagine
    for idx, img_name in enumerate(selected_images, 1):
        base_name = img_name.replace('_0000.png', '')
        print(f"[{idx}/{len(selected_images)}] {base_name}")
        
        img, pred, gt, error = load_image_triple(img_name)
        
        if error:
            print(f"  ⚠️  {error}")
            continue
        
        # Crea visualizzazione
        save_path = os.path.join(output_images_dir, f"{base_name}_comparison.png")
        dice, iou, accuracy = visualize_comparison(img, pred, gt, title=base_name, save_path=save_path)
        
        # Salva risultati
        results.append((img_name, dice, iou, accuracy))
    
    # Statistiche in console
    print("\n" + "="*70)
    print("📊 STATISTICHE FINALI")
    print("="*70)
    
    dices = [r[1] for r in results]
    ious = [r[2] for r in results]
    accs = [r[3] for r in results]
    
    print(f"Campioni elaborati: {len(results)}")
    print(f"\nDice Coefficient:")
    print(f"  Media: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  Range: [{np.min(dices):.4f}, {np.max(dices):.4f}]")
    print(f"\nIoU:")
    print(f"  Media: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"  Range: [{np.min(ious):.4f}, {np.max(ious):.4f}]")
    print(f"\nPixel Accuracy:")
    print(f"  Media: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Range: [{np.min(accs):.4f}, {np.max(accs):.4f}]")
    print("="*70)
    
    # Scrivi report su file
    write_report(results, output_report_file)
    
    print(f"\n✅ Elaborazione completata!")
    print(f"   📊 Report: {output_report_file}")
    print(f"   🖼️  Immagini: {output_images_dir}/\n")


if __name__ == "__main__":
    main()
