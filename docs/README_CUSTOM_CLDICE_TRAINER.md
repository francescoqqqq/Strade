# 🧠 Custom nnU-Net Trainer con clDice Loss

Implementazione della **clDice (Centerline Dice) Loss** per segmentazione di strutture tubolari (strade, vasi sanguigni) seguendo il paper:

> **"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"**

## 🎯 Obiettivo

La clDice loss migliora la **preservazione della topologia** nelle segmentazioni, riducendo:
- ❌ Disconnessioni nelle strade
- ❌ Buchi nelle strutture continue
- ❌ Errori di connettività

Risultato atteso: **Strade più continue e connesse** anche con leggero trade-off sul Dice standard.

---

## 📦 Componenti

```
custom_nnunet/
├── __init__.py                  # Package initialization
├── soft_skeleton.py             # Soft (differentiable) skeletonization
├── cldice_loss.py              # clDice loss implementation
└── trainer_cldice.py           # Custom nnU-Net trainer

test_cldice_implementation.py   # Test suite completa
README_CUSTOM_CLDICE_TRAINER.md # Questa guida
```

### 1. **Soft Skeletonization** (`soft_skeleton.py`)

Implementa l'**Algoritmo 1** del paper usando operazioni differenziabili:
- ✅ **Max pooling** → Dilatazione morfologica
- ✅ **Min pooling** (`-max_pool(-x)`) → Erosione morfologica
- ✅ **Iterativo** (k=5 iterazioni) → Estrae centerline progressivamente
- ✅ **Differenziabile** → Mantiene gradiente per backprop

### 2. **clDice Loss** (`cldice_loss.py`)

Implementa l'**Algoritmo 2** del paper:
- Estrae skeleton da predizione e ground truth
- Calcola **Topology Precision**: `|skel_pred ∩ mask_gt| / |skel_pred|`
- Calcola **Topology Sensitivity**: `|skel_gt ∩ mask_pred| / |skel_gt|`
- clDice = **media armonica** di T_prec e T_sens
- Loss = `1 - clDice` (per minimizzazione)

### 3. **Custom Trainer** (`trainer_cldice.py`)

Integra tutto in nnU-Net v2:
- Eredita da `nnUNetTrainer` standard
- Override solo del metodo `_build_loss`
- **Combined Loss** (Equazione 3 del paper):
  ```
  Loss = (1 - α) * (Dice + CE) + α * clDice
  ```
  dove α = 0.4 (default)

---

## 🚀 Uso

### Step 1: Test Implementazione

**Prima di trainare**, verifica che tutto funzioni:

```bash
cd /workspace
python test_cldice_implementation.py
```

Output atteso:
```
🧪 TEST SUITE: clDice Implementation
======================================================================
✅ PASSED  Soft Skeletonization
✅ PASSED  clDice Loss
✅ PASSED  Combined Loss
✅ PASSED  Edge Cases
✅ PASSED  Realistic Scenario

🎉 ALL TESTS PASSED!
✓ L'implementazione è pronta per il training nnU-Net
```

Se tutti i test passano → procedi!

---

### Step 2: Training con clDice

Usa il custom trainer invece di quello standard:

```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

Parametri:
- `001`: Dataset ID (Dataset001_Strade)
- `2d`: Configurazione 2D
- `0`: Fold number
- `-tr nnUNetTrainerClDice`: **IMPORTANTE** specifica il custom trainer

#### Output Directory

Il training salverà i risultati in:
```
/workspace/nnUNet_results/Dataset001_Strade/
└── nnUNetTrainerClDice__nnUNetPlans__2d/
    └── fold_0/
        ├── checkpoint_best.pth
        ├── checkpoint_final.pth
        └── validation/
```

⚠️ **Nota**: Directory diversa da baseline (`nnUNetTrainer` → `nnUNetTrainerClDice`)

---

### Step 3: Predizioni

Dopo il training, genera predizioni:

```bash
nnUNetv2_predict \
  -i /workspace/nnUNet_raw/Dataset001_Strade/imagesTr \
  -o /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
  -d 001 \
  -c 2d \
  -f 0 \
  -tr nnUNetTrainerClDice
```

⚠️ **Importante**: Aggiungi `-tr nnUNetTrainerClDice` per usare il custom trainer!

---

### Step 4: Confronto con Baseline

Usa lo script `test_predictions.py` modificato:

1. **Copia risultati nella struttura corretta**:
```bash
# Se necessario, sposta le predizioni
mkdir -p /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/
mv /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
   /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```

2. **Testa modalità confronto**:
```python
# In test_predictions.py, riga 35:
COMPARISON_MODE = 'both'  # Confronta baseline vs clDice
```

```bash
python test_predictions.py
```

Vedrai:
- 📊 Tabelle comparative con DELTA
- 🖼️ Immagini side-by-side
- 📈 Metriche clDice per entrambi i modelli

---

## ⚙️ Configurazione Avanzata

### Varianti del Trainer

Sono disponibili 3 varianti pre-configurate:

#### 1. **Standard** (α=0.4, k=5) ← Raccomandato
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

#### 2. **Alta priorità topologia** (α=0.6)
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_HighAlpha
```
Usa se vuoi massimizzare la connettività a scapito di un po' di Dice score.

#### 3. **Conservativo** (α=0.2)
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_LowAlpha
```
Usa se α=0.4 degrada troppo il Dice standard.

#### 4. **Strutture spesse** (k=10)
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_ThickStructures
```
Usa se le tue strade sono molto larghe (>10 pixel).

---

### Personalizzazione Manuale

Per creare la tua variante, modifica `trainer_cldice.py`:

```python
class nnUNetTrainerClDice_Custom(nnUNetTrainerClDice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldice_alpha = 0.5        # Peso clDice (0-1)
        self.cldice_iterations = 7     # Iterazioni skeleton (1-15)
        self.cldice_smooth = 1e-5      # Epsilon stabilità (1e-7 - 1e-4)
```

---

## 📊 Interpretazione Risultati

### Metriche da Monitorare

Durante il training, monitora:
- **Training Loss**: Dovrebbe diminuire stabilmente
- **Validation Dice**: Potrebbe essere 1-3% più basso del baseline (trade-off accettabile)
- **Validation clDice**: Dovrebbe migliorare rispetto al baseline

### Risultati Attesi

| Metrica | Baseline | Con clDice | DELTA |
|---------|----------|------------|-------|
| Dice | 0.912 | ~0.905-0.915 | -0.007 ~ +0.003 |
| IoU | 0.845 | ~0.840-0.850 | -0.005 ~ +0.005 |
| **clDice** | 0.876 | **~0.890-0.920** | **+0.014 ~ +0.044** 🎯 |

**Obiettivo principale**: ↑ clDice (migliore topologia)

### Valutazione Qualitativa

Confronta visivamente le immagini:
- ✅ **Meno disconnessioni** nelle strade lunghe
- ✅ **Meno buchi** nelle regioni continue
- ✅ **Migliore continuità** agli incroci
- ⚠️ Possibili margini leggermente meno precisi (trade-off)

---

## 🐛 Troubleshooting

### Problema: Training instabile / Loss NaN

**Causa**: Alpha troppo alto o smooth troppo basso

**Soluzione**:
```python
self.cldice_alpha = 0.2  # Riduci alpha
self.cldice_smooth = 1e-4  # Aumenta epsilon
```

### Problema: Nessun miglioramento clDice

**Causa**: Iterazioni troppo basse per la larghezza delle strade

**Soluzione**:
```python
self.cldice_iterations = 10  # Aumenta k
```

### Problema: Dice score degrada troppo

**Causa**: Alpha troppo alto (troppa priorità a topologia)

**Soluzione**:
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_LowAlpha  # α=0.2
```

### Problema: ImportError del custom trainer

**Causa**: Trainer non nel path di nnU-Net

**Soluzione**:
```bash
# Aggiungi al PYTHONPATH
export PYTHONPATH="/workspace:$PYTHONPATH"

# Oppure copia nella directory nnU-Net
cp -r custom_nnunet /path/to/nnunetv2/training/nnUNetTrainer/variants/
```

---

## 📚 Riferimenti

### Paper
- **clDice**: "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
- ArXiv: https://arxiv.org/abs/2003.07311

### Implementazione
- Basato su nnU-Net v2
- PyTorch differentiable operations
- Segue rigorosamente Algoritmi 1 e 2 del paper

---

## 🧪 Validazione Implementazione

Prima di usare in produzione, verifica:

✅ **Test Suite**: Tutti i test passano
```bash
python test_cldice_implementation.py
```

✅ **Training converge**: Loss diminuisce stabilmente

✅ **clDice migliora**: Confronto quantitativo con baseline

✅ **Qualità visiva**: Strade più connesse nelle visualizzazioni

---

## 🎓 Note Tecniche

### Differenze da Implementazioni Standard

1. **Soft vs Hard**: Usa operazioni differenziabili (no OpenCV/skimage)
2. **Iterativo**: Skeleton estratto progressivamente (no thinning one-shot)
3. **GPU-friendly**: Tutto in PyTorch per efficienza
4. **Batch-aware**: Gestisce batch di qualsiasi dimensione

### Complessità Computazionale

- **Tempo training**: +15-25% rispetto a baseline (dovuto a skeleton extraction)
- **Memoria**: +10-15% (skeleton maps aggiuntivi)
- **Inference**: Identica al baseline (loss non usata)

### Limitazioni

- Ottimizzato per strutture **tubolari/lineari** (strade, vasi)
- Meno efficace per regioni compatte (es. edifici)
- Trade-off Dice vs Topologia (non sempre win-win)

---

## 🚀 Quick Start

TL;DR - Comandi essenziali:

```bash
# 1. Test
python test_cldice_implementation.py

# 2. Train
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice

# 3. Predict
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 2d -f 0 -tr nnUNetTrainerClDice

# 4. Compare
python test_predictions.py  # COMPARISON_MODE = 'both'
```

---

**Buon training! 🎉**

Per domande o bug: controlla i test e i log di training.

