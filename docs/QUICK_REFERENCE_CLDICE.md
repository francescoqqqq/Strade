# ⚡ Quick Reference - clDice Training

## 🎯 Comandi Essenziali

### 1. Test (Opzionale)
```bash
python test_cldice_implementation.py
```
✅ Tutti i test devono passare (già verificato)

---

### 2. Training
```bash
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

**Varianti**:
```bash
# Alpha 0.6 (più topologia)
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_HighAlpha

# Alpha 0.2 (conservativo)
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_LowAlpha

# k=10 (strade spesse)
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice_ThickStructures
```

---

### 3. Predizioni
```bash
nnUNetv2_predict \
  -i /workspace/nnUNet_raw/Dataset001_Strade/imagesTr \
  -o /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
  -d 001 -c 2d -f 0 \
  -tr nnUNetTrainerClDice
```

---

### 4. Organizza Risultati
```bash
mkdir -p /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/

cp -r /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
      /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```

---

### 5. Confronto
```python
# In test_predictions.py, riga 35:
COMPARISON_MODE = 'both'
```

```bash
python test_predictions.py
```

---

## 📊 Output Directory

```
/workspace/nnUNet_results/Dataset001_Strade/
├── baseline_dice_only/                         # ✅ Già trainato
│   └── nnUNetTrainer__nnUNetPlans__2d/
│       └── fold_0/
│
├── nnUNetTrainerClDice__nnUNetPlans__2d/      # ⏳ Nuovo training
│   └── fold_0/
│       ├── checkpoint_best.pth
│       ├── checkpoint_final.pth
│       └── validation/
│
└── with_cldice_loss/                           # ⏳ Per confronto
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            └── validation/
```

---

## 🎛️ Parametri Chiave

| Parametro | Default | Quando Cambiare |
|-----------|---------|----------------|
| **alpha** | 0.4 | Se Dice degrada >5% → 0.2<br>Se vuoi max topologia → 0.6 |
| **k (iterations)** | 5 | Strade molto larghe → 10<br>Strade sottilissime → 3 |
| **smooth** | 1e-5 | Loss instabile → 1e-4 |

---

## ✅ Checklist

- [x] Test suite passa
- [x] Dataset in `nnUNet_raw/Dataset001_Strade/`
- [ ] GPU disponibile: `nvidia-smi`
- [ ] Spazio disco: ~10 GB liberi
- [ ] PYTHONPATH: `export PYTHONPATH="/workspace:$PYTHONPATH"`

---

## 📈 Metriche Attese

| Metrica | Baseline | Con clDice | Obiettivo |
|---------|----------|------------|-----------|
| Dice | 0.912 | ~0.910 | ≈ |
| **clDice** | 0.876 | **~0.900** | ↑ **+2-4%** |

---

## 🐛 Fix Rapidi

**Loss NaN?**
```python
self.cldice_smooth = 1e-4  # In trainer_cldice.py
```

**Dice peggiora troppo?**
```bash
nnUNetv2_train 001 2d 0 --trainer nnUNetTrainerClDice_LowAlpha
```

**clDice non migliora?**
```bash
nnUNetv2_train 001 2d 0 --trainer nnUNetTrainerClDice_ThickStructures
```

---

## 📚 Docs

- `README_CUSTOM_CLDICE_TRAINER.md` → Guida completa
- `IMPLEMENTAZIONE_CLDICE_COMPLETATA.md` → Riepilogo
- `test_cldice_implementation.py` → Test suite

---

**Ready to train! 🚀**

