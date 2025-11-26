# 📖 Guida test_predictions.py - Modalità Confronto

## 🎯 Panoramica

Lo script `test_predictions.py` ora supporta **3 modalità di test**:
1. **baseline_only**: Testa solo il modello baseline (Dice Loss)
2. **with_cldice_only**: Testa solo il modello con clDice Loss
3. **both**: Confronta entrambi i modelli side-by-side

## 🔧 Configurazione

### Variabile Principale (riga 24)

```python
COMPARISON_MODE = 'both'  # Opzioni: 'baseline_only', 'with_cldice_only', 'both'
```

### Altre Impostazioni

```python
# Modalità selezione immagini (riga 53)
TEST_MODE = 'random'      # 'all', 'random', 'specific'
NUM_RANDOM = 20           # Numero di immagini se mode='random'
SPECIFIC_IMAGES = [...]   # Lista numeri se mode='specific'
```

## 📁 Struttura Output

Lo script salva i risultati in directory diverse a seconda della modalità:

```
risultati/
├── baseline_dice_only/         # COMPARISON_MODE = 'baseline_only'
│   ├── Confronto_imm/
│   └── risultati_test.txt
│
├── with_cldice_loss/           # COMPARISON_MODE = 'with_cldice_only'
│   ├── Confronto_imm/
│   └── risultati_test.txt
│
└── confronto_both/             # COMPARISON_MODE = 'both'
    ├── Confronto_imm/          # Immagini con confronto side-by-side
    └── risultati_test.txt      # Report comparativo
```

## 🖼️ Output Visivo

### Modalità Singolo Modello ('baseline_only' o 'with_cldice_only')

Ogni immagine mostra 4 pannelli in orizzontale:
```
[Satellite Image] [Prediction] [Ground Truth] [Overlay]
```

**Metriche mostrate**: Dice, IoU, Accuracy, clDice

### Modalità Confronto ('both')

Ogni immagine mostra 2 righe x 4 colonne:
```
RIGA 1 (BASELINE):
[Satellite Image] [Prediction Baseline] [Ground Truth] [Overlay Baseline]

RIGA 2 (WITH CLDICE):
[Satellite Image] [Prediction clDice] [Ground Truth] [Overlay clDice]
```

**Metriche mostrate**: 
- Baseline: Dice, IoU, Accuracy, clDice
- With clDice: Dice, IoU, Accuracy, clDice
- **DELTA**: Differenza tra i due modelli

## 📊 Report Testuale

### Singolo Modello

```
═══════════════════════════════════════
STATISTICHE GLOBALI
═══════════════════════════════════════

Metrica               Media    Std Dev       Min        Max
─────────────────────────────────────────────────────────
Dice Coefficient      0.9123     0.0234    0.8567     0.9567
IoU                   0.8456     0.0345    0.7234     0.9123
...

═══════════════════════════════════════
METRICHE PER OGNI IMMAGINE
═══════════════════════════════════════

Immagine                 Dice        IoU   Accuracy     clDice
─────────────────────────────────────────────────────────
strade_0178          0.9234     0.8567     0.9856     0.8923
...
```

### Confronto Entrambi i Modelli

```
═══════════════════════════════════════════════════════════════
STATISTICHE GLOBALI - CONFRONTO TRA MODELLI
═══════════════════════════════════════════════════════════════

Metrica                   BASELINE                    WITH CLDICE                 DELTA
                    Media   Std   Min   Max      Media   Std   Min   Max
──────────────────────────────────────────────────────────────────────────────
Dice Coefficient   0.9123 0.023 0.856 0.956    0.9145 0.022 0.861 0.959    +0.0022
clDice (Topology)  0.8756 0.034 0.801 0.934    0.8923 0.031 0.823 0.945    +0.0167
...

═══════════════════════════════════════════════════════════════
METRICHE PER OGNI IMMAGINE - CONFRONTO
═══════════════════════════════════════════════════════════════

Immagine           BASELINE                        WITH CLDICE                   DELTA
               Dice   IoU   Acc clDice        Dice   IoU   Acc clDice      clDice
───────────────────────────────────────────────────────────────────────────────
strade_0178   0.923 0.856 0.985 0.892        0.925 0.859 0.986 0.905      +0.013
...
```

## 🚀 Esempi di Utilizzo

### Esempio 1: Testare Solo Baseline (già fatto)

```python
# In test_predictions.py, riga 24:
COMPARISON_MODE = 'baseline_only'
TEST_MODE = 'random'
NUM_RANDOM = 20
```

```bash
python test_predictions.py
```

**Output**: `risultati/baseline_dice_only/`

### Esempio 2: Testare Solo Modello con clDice (dopo training)

```python
# In test_predictions.py, riga 24:
COMPARISON_MODE = 'with_cldice_only'
TEST_MODE = 'random'
NUM_RANDOM = 20
```

```bash
python test_predictions.py
```

**Output**: `risultati/with_cldice_loss/`

### Esempio 3: Confrontare Entrambi i Modelli

```python
# In test_predictions.py, riga 24:
COMPARISON_MODE = 'both'
TEST_MODE = 'random'
NUM_RANDOM = 20
```

```bash
python test_predictions.py
```

**Output**: `risultati/confronto_both/`

### Esempio 4: Confronto su Immagini Specifiche

```python
COMPARISON_MODE = 'both'
TEST_MODE = 'specific'
SPECIFIC_IMAGES = [178, 302, 362, 390, 571]  # Numeri specifici
```

## ⚠️ Note Importanti

### Per Usare Modalità 'both'

Prima di usare `COMPARISON_MODE = 'both'`, assicurati che:
1. ✅ Il modello baseline sia trainato
2. ✅ Il modello con clDice sia trainato
3. ✅ Entrambi abbiano predizioni in `validation/`

Se manca uno dei due, lo script terminerà con errore.

### Requisiti Predizioni

Le predizioni devono esistere qui:
```
/workspace/nnUNet_results/Dataset001_Strade/
├── baseline_dice_only/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/
└── with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/
```

## 📈 Interpretazione DELTA (modalità 'both')

Nel report comparativo, il **DELTA** indica:
- **DELTA > 0**: Il modello con clDice è migliore
- **DELTA < 0**: Il modello baseline è migliore
- **DELTA ≈ 0**: Performance simili

### clDice DELTA è l'indicatore chiave!

Se hai trainato con clDice Loss, ci aspettiamo:
- **clDice DELTA**: positivo e significativo (+0.01 ~ +0.05)
- **Dice/IoU DELTA**: leggermente positivo o neutro (possibile trade-off)

## 🔄 Workflow Completo

1. **Baseline già testato** ✅
   ```python
   COMPARISON_MODE = 'baseline_only'
   ```

2. **Traini modello con clDice** ⏳
   ```bash
   # Dopo implementazione clDice Loss
   nnUNetv2_train ...
   ```

3. **Testi modello con clDice**
   ```python
   COMPARISON_MODE = 'with_cldice_only'
   ```

4. **Confronti entrambi**
   ```python
   COMPARISON_MODE = 'both'
   ```

5. **Analizzi differenze visive e numeriche**
   - Guarda immagini in `risultati/confronto_both/Confronto_imm/`
   - Leggi report in `risultati/confronto_both/risultati_test.txt`

---
*Aggiornato: 2025-11-25*
*Versione: 2.0 - Con supporto confronto side-by-side*

