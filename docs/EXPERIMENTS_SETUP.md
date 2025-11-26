# 🧪 Setup Esperimenti: Baseline vs clDice Loss

## 📁 Struttura Completa Organizzata

```
/workspace/
│
├── nnUNet_raw/
│   └── Dataset001_Strade/          # Dataset originale (immutato)
│       ├── imagesTr/
│       └── labelsTr/
│
├── nnUNet_results/
│   └── Dataset001_Strade/
│       ├── baseline_dice_only/     # ✅ Modello trainato con Dice Loss standard
│       │   └── nnUNetTrainer__nnUNetPlans__2d/
│       │       ├── fold_0/
│       │       │   ├── checkpoint_best.pth
│       │       │   ├── checkpoint_final.pth
│       │       │   ├── validation/        # Predizioni validation set
│       │       │   └── training_log_*.txt
│       │       └── plans.json
│       │
│       └── with_cldice_loss/       # ⏳ Modello da trainare con Dice + clDice Loss
│           └── nnUNetTrainer__nnUNetPlans__2d/
│               └── fold_0/
│
├── risultati/
│   ├── baseline_dice_only/         # ✅ Risultati test modello baseline
│   │   ├── Confronto_imm/          # 20 immagini con visualizzazioni
│   │   │   ├── strade_0178_comparison.png
│   │   │   ├── strade_0302_comparison.png
│   │   │   └── ... (18 altre)
│   │   └── risultati_test.txt      # Report con tutte le metriche
│   │
│   └── with_cldice_loss/           # ⏳ Risultati test modello con clDice
│       ├── Confronto_imm/          # Stesse 20 immagini per confronto
│       └── risultati_test.txt
│
└── test_predictions.py             # ✅ Script aggiornato con clDice metrica
```

## 🎯 Workflow degli Esperimenti

### Fase 1: Baseline (✅ COMPLETATO)
1. ✅ Training con Dice Loss standard
2. ✅ Test su validation set con metriche (Dice, IoU, Accuracy, clDice)
3. ✅ Risultati salvati in `risultati/baseline_dice_only/`

### Fase 2: Con clDice Loss (⏳ PROSSIMO STEP)
1. ⏳ Implementare clDice nella loss function di nnU-Net
2. ⏳ Trainare nuovo modello
3. ⏳ Testare sulle STESSE immagini del baseline
4. ⏳ Confrontare risultati side-by-side

## 🔧 Come Usare test_predictions.py

Lo script è stato aggiornato per supportare entrambi gli esperimenti:

### Testare Baseline:
```python
# In test_predictions.py, riga 24:
EXPERIMENT_NAME = 'baseline_dice_only'
```

### Testare modello con clDice (dopo training):
```python
# In test_predictions.py, riga 24:
EXPERIMENT_NAME = 'with_cldice_loss'
```

Poi esegui:
```bash
python test_predictions.py
```

## 📊 Metriche Valutate

Entrambi i modelli saranno valutati con:

| Metrica | Descrizione |
|---------|-------------|
| **Dice Coefficient** | Overlap generale tra predizione e GT |
| **IoU** | Intersection over Union |
| **Pixel Accuracy** | Accuratezza pixel-level |
| **clDice** | Preservazione topologia (centerline/connettività) |

## 🔄 Prossimi Passi

1. **Implementare clDice Loss** in nnU-Net
   - Creare custom trainer con combined loss: `Dice + α * clDice`
   - α = peso clDice (da tuning, es: 0.1-0.5)

2. **Training**
   ```bash
   nnUNetv2_train 001 2d 0 --trainer CustomTrainerWithClDice
   ```

3. **Test e Confronto**
   - Cambiare `EXPERIMENT_NAME` in `test_predictions.py`
   - Eseguire test sulle stesse 20 immagini
   - Confrontare visivamente e numericamente

## 📈 Confronto Atteso

**Ipotesi:**
- **Dice, IoU**: simili o leggermente inferiori (trade-off)
- **clDice**: miglioramento significativo (obiettivo principale)
- **Qualitativo**: meno disconnessioni, strade più continue

## 📝 Note
- Tutti i path sono configurati automaticamente
- I risultati baseline sono preservati e immutabili
- Facile switch tra esperimenti cambiando solo `EXPERIMENT_NAME`

---
*Organizzato: 2025-11-25*
*Status: Pronto per implementazione clDice Loss*

