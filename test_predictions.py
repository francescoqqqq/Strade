#!/usr/bin/env python3
"""
Script per testare le predizioni di nnU-Net su Dataset001_Strade

MODALITÀ:
  - baseline_only: Testa solo modello baseline (Dice Loss)
  - with_cldice_only: Testa solo modello con clDice Loss
  - both: Confronta entrambi i modelli side-by-side

OUTPUT:
  - Visualizzazioni: risultati/<modalità>/Confronto_imm/
  - Report testuale: risultati/<modalità>/risultati_test.txt
  
METRICHE:
  - Dice Coefficient, IoU, Pixel Accuracy, clDice (topologia)
"""

import os
import sys
import numpy as np
from PIL import Image  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import random
from datetime import datetime
from skimage.morphology import skeletonize  # type: ignore

# ========== CONFIGURAZIONE ==========
dataset_name = "Dataset001_Strade"
dataset_id = "001"
fold = 0

# ========== MODALITÀ CONFRONTO ==========
# Opzioni: 'baseline_only', 'with_cldice_only', 'both'
COMPARISON_MODE = 'baseline_only'
# - 'baseline_only': testa solo modello baseline
# - 'with_cldice_only': testa solo modello con clDice
# - 'both': confronta entrambi i modelli side-by-side


# Directory
raw_data_dir = f"/workspace/nnUNet_raw/{dataset_name}"
images_dir = os.path.join(raw_data_dir, "imagesTr")
labels_dir = os.path.join(raw_data_dir, "labelsTr")

# Directory predizioni (dipendono dalla modalità)
base_results_dir = f"/workspace/nnUNet_results/{dataset_name}"
predictions_dir_baseline = os.path.join(base_results_dir, "baseline_dice_only/nnUNetTrainer__nnUNetPlans__2d/fold_{fold}/validation")
predictions_dir_cldice = os.path.join(base_results_dir, "with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_{fold}/validation")

# Directory output organizzata
if COMPARISON_MODE == 'both':
    output_base_dir = "/workspace/risultati/confronto_both"
elif COMPARISON_MODE == 'baseline_only':
    output_base_dir = "/workspace/risultati/baseline_dice_only"
else:
    output_base_dir = "/workspace/risultati/with_cldice_loss"

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
    dirs_to_check = []
    
    if COMPARISON_MODE in ['baseline_only', 'both']:
        dirs_to_check.append(('baseline', predictions_dir_baseline))
    if COMPARISON_MODE in ['with_cldice_only', 'both']:
        dirs_to_check.append(('with_cldice', predictions_dir_cldice))
    
    for name, pred_dir in dirs_to_check:
        if not os.path.exists(pred_dir) or len(os.listdir(pred_dir)) == 0:
            print(f"\n🔮 Le predizioni per '{name}' non esistono ancora. Eseguile con:\n")
            print(f"nnUNetv2_predict -i {images_dir} -o {pred_dir} -d {dataset_id} -c 2d -f {fold}\n")
            print(f"⚠️  Esegui prima le predizioni per '{name}', poi rilancia questo script.\n")
            sys.exit(0)


def load_image_data(image_name):
    """
    Carica immagine satellitare, predizione(i) e ground truth.
    
    Returns:
        Se COMPARISON_MODE = 'both': (img, pred_baseline, pred_cldice, gt, error)
        Altrimenti: (img, pred, gt, error)
    """
    base_name = image_name.replace('_0000.png', '')
    
    # Percorsi comuni
    img_path = os.path.join(images_dir, image_name)
    gt_path = os.path.join(labels_dir, base_name + '.png')
    
    # Verifica esistenza
    if not os.path.exists(img_path):
        error_msg = f"Immagine non trovata: {img_path}"
        if COMPARISON_MODE == 'both':
            return None, None, None, None, error_msg
        else:
            return None, None, None, error_msg
    
    if not os.path.exists(gt_path):
        error_msg = f"Ground truth non trovata: {gt_path}"
        if COMPARISON_MODE == 'both':
            return None, None, None, None, error_msg
        else:
            return None, None, None, error_msg
    
    # Carica immagine e GT
    img = np.array(Image.open(img_path))
    gt = np.array(Image.open(gt_path))
    
    if COMPARISON_MODE == 'both':
        # Carica entrambe le predizioni
        pred_path_baseline = os.path.join(predictions_dir_baseline, base_name + '.png')
        pred_path_cldice = os.path.join(predictions_dir_cldice, base_name + '.png')
        
        if not os.path.exists(pred_path_baseline):
            return None, None, None, None, f"Predizione baseline non trovata: {pred_path_baseline}"
        if not os.path.exists(pred_path_cldice):
            return None, None, None, None, f"Predizione clDice non trovata: {pred_path_cldice}"
        
        pred_baseline = np.array(Image.open(pred_path_baseline))
        pred_cldice = np.array(Image.open(pred_path_cldice))
        
        return img, pred_baseline, pred_cldice, gt, None
    
    else:
        # Carica solo una predizione
        pred_dir = predictions_dir_baseline if COMPARISON_MODE == 'baseline_only' else predictions_dir_cldice
        pred_path = os.path.join(pred_dir, base_name + '.png')
        
        if not os.path.exists(pred_path):
            return None, None, None, f"Predizione non trovata: {pred_path}"
        
        pred = np.array(Image.open(pred_path))
        return img, pred, gt, None


def compute_skeleton(mask):
    """
    Calcola lo scheletro morfologico di una maschera binaria 2D.
    
    Args:
        mask: numpy array binario (valori 0/1 o 0/255)
    
    Returns:
        skeleton: numpy array binario dello scheletro
    """
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool)
    return skeletonize(mask_bool)


def calculate_cldice(pred, gt):
    """
    Calcola centerline Dice (clDice) coefficient.
    
    La metrica clDice valuta quanto bene la topologia (centerline) delle predizioni
    corrisponde alla ground truth.
    
    Args:
        pred: numpy array della predizione (valori 0/1 o 0/255)
        gt: numpy array della ground truth (valori 0/1 o 0/255)
    
    Returns:
        cldice: float [0, 1], dove 1 = perfetta corrispondenza topologica
    """
    pred_bool = pred > 0
    gt_bool = gt > 0
    
    # Calcola scheletri
    skel_pred = compute_skeleton(pred_bool)
    skel_gt = compute_skeleton(gt_bool)
    
    # Se entrambi gli scheletri sono vuoti, ritorna 1 (match perfetto di vuoti)
    if not np.any(skel_pred) and not np.any(skel_gt):
        return 1.0
    
    # Se solo uno è vuoto, ritorna 0
    if not np.any(skel_pred) or not np.any(skel_gt):
        return 0.0
    
    # Topology Precision: quanto dello scheletro predetto è dentro la GT
    tprec_numerator = np.sum(skel_pred & gt_bool)
    tprec_denominator = np.sum(skel_pred)
    tprec = tprec_numerator / tprec_denominator if tprec_denominator > 0 else 0.0
    
    # Topology Sensitivity: quanto dello scheletro GT è coperto dalla predizione
    tsens_numerator = np.sum(skel_gt & pred_bool)
    tsens_denominator = np.sum(skel_gt)
    tsens = tsens_numerator / tsens_denominator if tsens_denominator > 0 else 0.0
    
    # clDice: media armonica di Tprec e Tsens
    if tprec + tsens == 0:
        return 0.0
    
    cldice = 2.0 * (tprec * tsens) / (tprec + tsens)
    
    return cldice


def calculate_metrics(pred, gt):
    """Calcola Dice, IoU, Accuracy e clDice"""
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
    
    # clDice (centerline Dice)
    cldice = calculate_cldice(pred, gt)
    
    return dice, iou, accuracy, cldice


def visualize_single_model(img, pred, gt, title="", model_name="", save_path=None):
    """Visualizza confronto per un singolo modello: Immagine | Predizione | Ground Truth | Overlay"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Immagine satellitare
    axes[0].imshow(img)
    axes[0].set_title('Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Predizione
    axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Prediction\n({model_name})', fontsize=12, fontweight='bold')
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
    dice, iou, accuracy, cldice = calculate_metrics(pred, gt)
    
    # Titolo con metriche
    fig.suptitle(f'{title}\nDice: {dice:.4f} | IoU: {iou:.4f} | Accuracy: {accuracy:.4f} | clDice: {cldice:.4f}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Salvato: {os.path.basename(save_path)}")
    
    plt.close(fig)
    
    return dice, iou, accuracy, cldice


def visualize_both_models(img, pred_baseline, pred_cldice, gt, title="", save_path=None):
    """Visualizza confronto tra entrambi i modelli"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    
    # ===== RIGA 1: BASELINE =====
    # Immagine satellitare
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Satellite Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Predizione Baseline
    axes[0, 1].imshow(pred_baseline, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Prediction\n(Baseline - Dice Only)', fontsize=11, fontweight='bold', color='#1f77b4')
    axes[0, 1].axis('off')
    
    # Ground Truth
    axes[0, 2].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Overlay Baseline
    overlay_baseline = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    overlay_baseline[gt > 0] = [255, 0, 0]
    overlay_baseline[pred_baseline > 0] = [0, 255, 0]
    overlap_baseline = (gt > 0) & (pred_baseline > 0)
    overlay_baseline[overlap_baseline] = [255, 255, 0]
    overlay_final_baseline = (0.6 * img + 0.4 * overlay_baseline).astype(np.uint8)
    axes[0, 3].imshow(overlay_final_baseline)
    axes[0, 3].set_title('Overlay Baseline\n(Red=GT, Green=Pred, Yellow=Match)', fontsize=10, fontweight='bold')
    axes[0, 3].axis('off')
    
    # ===== RIGA 2: WITH CLDICE =====
    # Immagine satellitare (ripetuta)
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Satellite Image', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Predizione clDice
    axes[1, 1].imshow(pred_cldice, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction\n(With clDice Loss)', fontsize=11, fontweight='bold', color='#ff7f0e')
    axes[1, 1].axis('off')
    
    # Ground Truth (ripetuta)
    axes[1, 2].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Overlay clDice
    overlay_cldice = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    overlay_cldice[gt > 0] = [255, 0, 0]
    overlay_cldice[pred_cldice > 0] = [0, 255, 0]
    overlap_cldice = (gt > 0) & (pred_cldice > 0)
    overlay_cldice[overlap_cldice] = [255, 255, 0]
    overlay_final_cldice = (0.6 * img + 0.4 * overlay_cldice).astype(np.uint8)
    axes[1, 3].imshow(overlay_final_cldice)
    axes[1, 3].set_title('Overlay clDice\n(Red=GT, Green=Pred, Yellow=Match)', fontsize=10, fontweight='bold')
    axes[1, 3].axis('off')
    
    # Calcola metriche per entrambi
    dice_b, iou_b, acc_b, cldice_b = calculate_metrics(pred_baseline, gt)
    dice_c, iou_c, acc_c, cldice_c = calculate_metrics(pred_cldice, gt)
    
    # Titolo con confronto metriche
    title_text = f'{title}\n'
    title_text += f'BASELINE: Dice={dice_b:.4f} | IoU={iou_b:.4f} | Acc={acc_b:.4f} | clDice={cldice_b:.4f}\n'
    title_text += f'WITH CLDICE: Dice={dice_c:.4f} | IoU={iou_c:.4f} | Acc={acc_c:.4f} | clDice={cldice_c:.4f}\n'
    title_text += f'DELTA: Dice={dice_c-dice_b:+.4f} | IoU={iou_c-iou_b:+.4f} | Acc={acc_c-acc_b:+.4f} | clDice={cldice_c-cldice_b:+.4f}'
    
    fig.suptitle(title_text, fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Salvato: {os.path.basename(save_path)}")
    
    plt.close(fig)
    
    return (dice_b, iou_b, acc_b, cldice_b), (dice_c, iou_c, acc_c, cldice_c)


def write_report(results, report_file):
    """Scrive report con tabelle metriche"""
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("═" * 120 + "\n")
        f.write("  REPORT RISULTATI TEST - SEGMENTAZIONE STRADE\n")
        f.write("═" * 120 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modalità confronto: {COMPARISON_MODE}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Numero campioni: {len(results)}\n")
        f.write(f"Checkpoint: checkpoint_best.pth\n")
        f.write("\n")
        
        # Determina se siamo in modalità confronto o singolo
        is_comparison = COMPARISON_MODE == 'both'
        
        if is_comparison:
            # Estrai metriche per entrambi i modelli
            dices_b = [r[1][0] for r in results]
            ious_b = [r[1][1] for r in results]
            accs_b = [r[1][2] for r in results]
            cldices_b = [r[1][3] for r in results]
            
            dices_c = [r[2][0] for r in results]
            ious_c = [r[2][1] for r in results]
            accs_c = [r[2][2] for r in results]
            cldices_c = [r[2][3] for r in results]
        else:
            # Singolo modello
            dices = [r[1] for r in results]
            ious = [r[2] for r in results]
            accs = [r[3] for r in results]
            cldices = [r[4] for r in results]
        
        # ═══ PRIMA: STATISTICHE GLOBALI ═══
        if is_comparison:
            f.write("═" * 120 + "\n")
            f.write("STATISTICHE GLOBALI - CONFRONTO TRA MODELLI\n")
            f.write("═" * 120 + "\n\n")
            
            f.write(f"{'Metrica':<20} {'BASELINE':^48} {'WITH CLDICE':^48}\n")
            f.write(f"{'':20} {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}   {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}   {'DELTA':>10}\n")
            f.write("─" * 120 + "\n")
            
            f.write(f"{'Dice Coefficient':<20} "
                    f"{np.mean(dices_b):>10.4f} {np.std(dices_b):>10.4f} {np.min(dices_b):>10.4f} {np.max(dices_b):>10.4f}   "
                    f"{np.mean(dices_c):>10.4f} {np.std(dices_c):>10.4f} {np.min(dices_c):>10.4f} {np.max(dices_c):>10.4f}   "
                    f"{np.mean(dices_c)-np.mean(dices_b):>+10.4f}\n")
            
            f.write(f"{'IoU':<20} "
                    f"{np.mean(ious_b):>10.4f} {np.std(ious_b):>10.4f} {np.min(ious_b):>10.4f} {np.max(ious_b):>10.4f}   "
                    f"{np.mean(ious_c):>10.4f} {np.std(ious_c):>10.4f} {np.min(ious_c):>10.4f} {np.max(ious_c):>10.4f}   "
                    f"{np.mean(ious_c)-np.mean(ious_b):>+10.4f}\n")
            
            f.write(f"{'Pixel Accuracy':<20} "
                    f"{np.mean(accs_b):>10.4f} {np.std(accs_b):>10.4f} {np.min(accs_b):>10.4f} {np.max(accs_b):>10.4f}   "
                    f"{np.mean(accs_c):>10.4f} {np.std(accs_c):>10.4f} {np.min(accs_c):>10.4f} {np.max(accs_c):>10.4f}   "
                    f"{np.mean(accs_c)-np.mean(accs_b):>+10.4f}\n")
            
            f.write(f"{'clDice (Topology)':<20} "
                    f"{np.mean(cldices_b):>10.4f} {np.std(cldices_b):>10.4f} {np.min(cldices_b):>10.4f} {np.max(cldices_b):>10.4f}   "
                    f"{np.mean(cldices_c):>10.4f} {np.std(cldices_c):>10.4f} {np.min(cldices_c):>10.4f} {np.max(cldices_c):>10.4f}   "
                    f"{np.mean(cldices_c)-np.mean(cldices_b):>+10.4f}\n")
            
            f.write("\n")
            
        else:
            # Singolo modello
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
            
            f.write(f"{'clDice (Topology)':<20} "
                    f"{np.mean(cldices):>12.4f} "
                    f"{np.std(cldices):>12.4f} "
                    f"{np.min(cldices):>10.4f} "
                    f"{np.max(cldices):>10.4f}\n")
            
            f.write("\n")
        
        # ═══ POI: TABELLA RISULTATI INDIVIDUALI ═══
        if is_comparison:
            f.write("═" * 120 + "\n")
            f.write("METRICHE PER OGNI IMMAGINE - CONFRONTO\n")
            f.write("═" * 120 + "\n\n")
            
            f.write(f"{'Immagine':<15} {'BASELINE':^44} {'WITH CLDICE':^44} {'DELTA':^15}\n")
            f.write(f"{'':15} {'Dice':>10} {'IoU':>10} {'Acc':>10} {'clDice':>10}  {'Dice':>10} {'IoU':>10} {'Acc':>10} {'clDice':>10}  {'clDice':>10}\n")
            f.write("─" * 120 + "\n")
            
            for img_name, metrics_b, metrics_c in results:
                base_name = img_name.replace('_0000.png', '')
                dice_b, iou_b, acc_b, cldice_b = metrics_b
                dice_c, iou_c, acc_c, cldice_c = metrics_c
                delta_cldice = cldice_c - cldice_b
                
                f.write(f"{base_name:<15} "
                       f"{dice_b:>10.4f} {iou_b:>10.4f} {acc_b:>10.4f} {cldice_b:>10.4f}  "
                       f"{dice_c:>10.4f} {iou_c:>10.4f} {acc_c:>10.4f} {cldice_c:>10.4f}  "
                       f"{delta_cldice:>+10.4f}\n")
            
            f.write("─" * 120 + "\n\n")
            
            # Performance breakdown - per entrambi i modelli
            f.write("─" * 120 + "\n")
            f.write("DISTRIBUZIONE PERFORMANCE (Dice Coefficient)\n")
            f.write("─" * 120 + "\n\n")
            
            exc_b = sum(1 for d in dices_b if d > 0.95)
            good_b = sum(1 for d in dices_b if 0.85 <= d <= 0.95)
            fair_b = sum(1 for d in dices_b if 0.70 <= d < 0.85)
            poor_b = sum(1 for d in dices_b if d < 0.70)
            
            exc_c = sum(1 for d in dices_c if d > 0.95)
            good_c = sum(1 for d in dices_c if 0.85 <= d <= 0.95)
            fair_c = sum(1 for d in dices_c if 0.70 <= d < 0.85)
            poor_c = sum(1 for d in dices_c if d < 0.70)
            
            total = len(dices_b)
            
            f.write(f"{'Categoria':<30} {'BASELINE':>20} {'WITH CLDICE':>20}\n")
            f.write(f"{'Eccellenti (Dice > 0.95)':<30} {exc_b:3d}/{total} ({exc_b/total*100:5.1f}%)    {exc_c:3d}/{total} ({exc_c/total*100:5.1f}%)\n")
            f.write(f"{'Buone (Dice 0.85-0.95)':<30} {good_b:3d}/{total} ({good_b/total*100:5.1f}%)    {good_c:3d}/{total} ({good_c/total*100:5.1f}%)\n")
            f.write(f"{'Discrete (Dice 0.70-0.85)':<30} {fair_b:3d}/{total} ({fair_b/total*100:5.1f}%)    {fair_c:3d}/{total} ({fair_c/total*100:5.1f}%)\n")
            f.write(f"{'Problematiche (Dice < 0.70)':<30} {poor_b:3d}/{total} ({poor_b/total*100:5.1f}%)    {poor_c:3d}/{total} ({poor_c/total*100:5.1f}%)\n")
            
            f.write("\n" + "═" * 120 + "\n")
            f.write(f"File immagini salvati in: {output_images_dir}/\n")
            f.write("═" * 120 + "\n")
            
        else:
            # Singolo modello
            f.write("═" * 80 + "\n")
            f.write("METRICHE PER OGNI IMMAGINE\n")
            f.write("═" * 80 + "\n\n")
            
            f.write(f"{'Immagine':<20} {'Dice':>10} {'IoU':>10} {'Accuracy':>10} {'clDice':>10}\n")
            f.write("─" * 80 + "\n")
            
            for img_name, dice, iou, acc, cldice in results:
                base_name = img_name.replace('_0000.png', '')
                f.write(f"{base_name:<20} {dice:>10.4f} {iou:>10.4f} {acc:>10.4f} {cldice:>10.4f}\n")
            
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
            f.write(f"File immagini salvati in: {output_images_dir}/\n")
            f.write("═" * 80 + "\n")
    
    print(f"\n✅ Report salvato in: {report_file}")


def main():
    print("\n" + "="*70)
    print("🔍 VISUALIZZAZIONE PREDIZIONI vs GROUND TRUTH")
    print("="*70)
    print(f"📦 Modalità confronto: {COMPARISON_MODE}")
    print("="*70 + "\n")
    
    # Verifica predizioni
    run_predictions_if_needed()
    
    # Crea directory output
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    print(f"📁 Output directory: {output_base_dir}/")
    print(f"   - Immagini: {output_images_dir}/")
    print(f"   - Report: {output_report_file}\n")
    
    # Determina directory da cui leggere le predizioni
    if COMPARISON_MODE == 'both':
        # Per 'both', usa baseline come riferimento per la lista immagini
        reference_dir = predictions_dir_baseline
    elif COMPARISON_MODE == 'baseline_only':
        reference_dir = predictions_dir_baseline
    else:
        reference_dir = predictions_dir_cldice
    
    # Lista immagini (solo quelle con predizioni nel validation set)
    pred_files = sorted([f for f in os.listdir(reference_dir) if f.endswith('.png')])
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
    if COMPARISON_MODE == 'both':
        # Modalità confronto: carica entrambe le predizioni
        for idx, img_name in enumerate(selected_images, 1):
            base_name = img_name.replace('_0000.png', '')
            print(f"[{idx}/{len(selected_images)}] {base_name}")
            
            data = load_image_data(img_name)
            if data[-1] is not None:  # Errore
                print(f"  ⚠️  {data[-1]}")
                continue
            
            img, pred_baseline, pred_cldice, gt = data[:-1]
            
            # Crea visualizzazione confronto
            save_path = os.path.join(output_images_dir, f"{base_name}_comparison.png")
            metrics_baseline, metrics_cldice = visualize_both_models(
                img, pred_baseline, pred_cldice, gt, 
                title=base_name, save_path=save_path
            )
            
            # Salva risultati
            results.append((img_name, metrics_baseline, metrics_cldice))
    
    else:
        # Modalità singolo modello
        model_name = 'Baseline' if COMPARISON_MODE == 'baseline_only' else 'With clDice'
        
        for idx, img_name in enumerate(selected_images, 1):
            base_name = img_name.replace('_0000.png', '')
            print(f"[{idx}/{len(selected_images)}] {base_name}")
            
            data = load_image_data(img_name)
            if data[-1] is not None:  # Errore
                print(f"  ⚠️  {data[-1]}")
                continue
            
            img, pred, gt = data[:-1]
            
            # Crea visualizzazione singola
            save_path = os.path.join(output_images_dir, f"{base_name}_comparison.png")
            dice, iou, accuracy, cldice = visualize_single_model(
                img, pred, gt, 
                title=base_name, model_name=model_name, save_path=save_path
            )
            
            # Salva risultati
            results.append((img_name, dice, iou, accuracy, cldice))
    
    # Statistiche in console
    print("\n" + "="*70)
    print("📊 STATISTICHE FINALI")
    print("="*70)
    
    if COMPARISON_MODE == 'both':
        # Estrai metriche per entrambi
        dices_b = [r[1][0] for r in results]
        cldices_b = [r[1][3] for r in results]
        dices_c = [r[2][0] for r in results]
        cldices_c = [r[2][3] for r in results]
        
        print(f"Campioni elaborati: {len(results)}\n")
        print(f"BASELINE:")
        print(f"  Dice: {np.mean(dices_b):.4f} ± {np.std(dices_b):.4f}")
        print(f"  clDice: {np.mean(cldices_b):.4f} ± {np.std(cldices_b):.4f}")
        print(f"\nWITH CLDICE:")
        print(f"  Dice: {np.mean(dices_c):.4f} ± {np.std(dices_c):.4f}")
        print(f"  clDice: {np.mean(cldices_c):.4f} ± {np.std(cldices_c):.4f}")
        print(f"\nDELTA:")
        print(f"  Dice: {np.mean(dices_c) - np.mean(dices_b):+.4f}")
        print(f"  clDice: {np.mean(cldices_c) - np.mean(cldices_b):+.4f}")
    else:
        # Singolo modello
        dices = [r[1] for r in results]
        ious = [r[2] for r in results]
        accs = [r[3] for r in results]
        cldices = [r[4] for r in results]
        
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
        print(f"\nclDice (Topology):")
        print(f"  Media: {np.mean(cldices):.4f} ± {np.std(cldices):.4f}")
        print(f"  Range: [{np.min(cldices):.4f}, {np.max(cldices):.4f}]")
    
    print("="*70)
    
    # Scrivi report su file
    write_report(results, output_report_file)
    
    print(f"\n✅ Elaborazione completata!")
    print(f"   📊 Report: {output_report_file}")
    print(f"   🖼️  Immagini: {output_images_dir}/\n")


if __name__ == "__main__":
    main()
