# 🚀 Avvia Training clDice - READY

## ✅ Fix Applicato

Ho configurato il trainer per **ignorare automaticamente** gli errori di `torch.compile` e fare fallback a eager mode.

**Cosa ho fatto**:
- ✅ Aggiunto `torch._dynamo.config.suppress_errors = True` nel trainer
- ✅ Pulito file parziali del tentativo fallito
- ✅ Trainer ora gestisce automaticamente il fallback

---

## 🚀 Comando Training (in tmux)

```bash
# 1. Avvia tmux
tmux new -s training_cldice

# 2. Lancia training (ora funzionerà!)
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice

# 3. Staccati: Ctrl+B poi D
```

---

## 📊 Cosa Aspettarsi

All'avvio vedrai:
```
======================================================================
🔬 Custom Trainer: nnUNetTrainerClDice
======================================================================
Loss Configuration:
  - Base Loss: Dice + Cross Entropy
  - Topology Loss: clDice (soft skeletonization)
  - Alpha (clDice weight): 0.4
  - Skeleton iterations (k): 5
  - Combined: Loss = 0.6 * (Dice+CE) + 0.4 * clDice
======================================================================

Using torch.compile...
```

Potrebbe vedere un warning su torch.compile che fallisce, ma **il training continuerà automaticamente** in eager mode! ✅

---

## 🎯 Durata

- **~2-4 ore** per completare
- Dipende da GPU e numero di epoche

---

## 📈 Monitoraggio

### Durante il Training (staccato da tmux)

```bash
# Riattacca a tmux
tmux a -t training_cldice

# Leggi log
tail -f /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/training_log_*.txt

# Controlla GPU
watch -n 2 nvidia-smi
```

---

## 🐛 Se Serve Aiuto

### Training si blocca?
```bash
# Termina sessione tmux
tmux kill-session -t training_cldice

# Pulizia
rm -rf /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/

# Rilancia
tmux new -s training_cldice
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

---

## ✨ Dopo il Training

1. **Predizioni**:
```bash
nnUNetv2_predict \
  -i /workspace/nnUNet_raw/Dataset001_Strade/imagesTr \
  -o /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
  -d 001 -c 2d -f 0 \
  -tr nnUNetTrainerClDice
```

2. **Organizza per confronto**:
```bash
mkdir -p /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/

cp -r /workspace/nnUNet_results/Dataset001_Strade/nnUNetTrainerClDice__nnUNetPlans__2d/fold_0/validation \
      /workspace/nnUNet_results/Dataset001_Strade/with_cldice_loss/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```

3. **Confronto**:
```python
# In test_predictions.py:
COMPARISON_MODE = 'both'
```
```bash
python test_predictions.py
```

---

**PRONTO PER PARTIRE! 🚀**

```bash
tmux new -s training_cldice
nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

