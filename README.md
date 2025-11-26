# 🛣️ Road Segmentation with nnU-Net + clDice Loss

**PhD Project - Francesco Girardello**

Segmentazione automatica di strade da immagini satellitari ad alta risoluzione usando **nnU-Net** con **loss topology-preserving (clDice)**.

---

## 🎯 Caratteristiche Principali

- ✅ Segmentazione strade da immagini satellitari (512×512 px)
- ✅ Dataset generato da OpenStreetMap + ESRI World Imagery
- ✅ **Custom clDice Loss** per preservare la topologia delle strade
- ✅ Confronto baseline vs clDice loss
- ✅ Metriche: Dice, IoU, Accuracy, **clDice**

---

## 📁 Struttura Progetto

```
/workspace/
├── 📄 README.md                      # Questo file
├── 📄 requirements.txt               # Dipendenze Python
│
├── 📁 custom_nnunet/                 # ⭐ Custom trainer con clDice loss
│   ├── __init__.py
│   ├── soft_skeleton.py              # Soft skeletonization (differenziabile)
│   ├── cldice_loss.py                # clDice loss implementation
│   └── trainer_cldice.py             # nnUNetTrainerClDice
│
├── 📁 configs/                       # Configurazioni
│   ├── config.yaml                   # Config generazione dataset
│   └── CONFIGURAZIONE_TRAINING.txt   # Parametri training
│
├── 📁 docs/                          # ⭐ Documentazione (TUTTI i .md vanno qui!)
│   ├── README.md                     # Documentazione completa
│   ├── QUICK_START.md                # Guida rapida
│   ├── INSTALL.md                    # Installazione
│   ├── HOW_TO_USE_TMUX.md           # Guida tmux
│   ├── README_CUSTOM_CLDICE_TRAINER.md  # Dettagli trainer custom
│   └── ...                           # Altri documenti
│
├── 📄 made_dataset.py                # Script generazione dataset
├── 📄 test_predictions.py            # ⭐ Test e confronto predizioni
├── 📄 visualize_samples.py           # Visualizzazione dataset
├── 📄 analyze_problematic_samples.py # Analisi campioni difficili
│
├── 📁 nnUNet_raw/                    # Dataset raw
│   └── Dataset001_Strade/
│       ├── imagesTr/                 # 2000 immagini RGB satellitari
│       ├── labelsTr/                 # Maschere binarie strade
│       └── dataset.json
│
├── 📁 nnUNet_preprocessed/           # Dati preprocessati per training
│   └── Dataset001_Strade/
│       └── nnUNetPlans_2d/
│
├── 📁 nnUNet_results/                # Risultati training
│   └── Dataset001_Strade/
│       ├── baseline_dice_only/       # Training con Dice+CE standard
│       ├── nnUNetTrainerClDice__nnUNetPlans__2d/  # Training con clDice
│       └── with_cldice_loss/         # (per organizzazione futura)
│
└── 📁 risultati/                     # Risultati test predizioni
    ├── baseline_dice_only/           # Test baseline
    ├── with_cldice_loss/             # Test clDice
    └── confronto_both/               # Confronto side-by-side
```

---

## 🚀 Quick Start

### 1️⃣ **Setup Ambiente**

```bash
# Installa dipendenze
pip install -r requirements.txt

# Configura percorsi nnU-Net
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
```

### 2️⃣ **Genera Dataset**

```bash
python made_dataset.py
# Output: 2000 immagini 512×512 in nnUNet_raw/Dataset001_Strade/
```

### 3️⃣ **Preprocessing**

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

### 4️⃣ **Training**

**Baseline (Dice + CE):**
```bash
nnUNetv2_train 001 2d 0
```

**Con clDice Loss (Raccomandato):**
```bash
nnUNet_compile=False nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

💡 **Tip:** Usa `tmux` per training lunghi (vedi `docs/HOW_TO_USE_TMUX.md`)

### 5️⃣ **Test Predizioni**

```bash
# Modifica COMPARISON_MODE in test_predictions.py:
# - 'baseline_only': Solo baseline
# - 'with_cldice_only': Solo clDice
# - 'both': Confronto side-by-side

python test_predictions.py
# Output: risultati/ con immagini confronto e report metriche
```

---

## 🧪 clDice Loss - Topology Preservation

La **clDice loss** è stata implementata per migliorare la preservazione della topologia delle strade (connettività):

### Componenti:
- **Soft Skeletonization:** Estrazione differenziabile dello scheletro morfologico
- **Topology Precision & Sensitivity:** Metriche per valutare connettività
- **Combined Loss:** `(1-α)·(Dice+CE) + α·clDice` con α=0.4

### File Coinvolti:
- `custom_nnunet/soft_skeleton.py` - Algoritmo skeletonization
- `custom_nnunet/cldice_loss.py` - Implementazione loss
- `custom_nnunet/trainer_cldice.py` - Trainer custom nnU-Net

### Documentazione:
Vedi `docs/README_CUSTOM_CLDICE_TRAINER.md` per dettagli implementativi.

---

## 📊 Metriche di Valutazione

Il testing (`test_predictions.py`) calcola:

| Metrica | Descrizione |
|---------|-------------|
| **Dice** | Sovrapposizione regioni (standard) |
| **IoU** | Intersection over Union |
| **Accuracy** | Accuratezza pixel-wise |
| **clDice** ⭐ | Preservazione topologia (connettività) |

---

## 📚 Documentazione

**📖 Principale:**
- [Documentazione Completa](docs/README.md)
- [Quick Start](docs/QUICK_START.md)
- [Installazione](docs/INSTALL.md)

**🔧 Training & clDice:**
- [Custom Trainer clDice](docs/README_CUSTOM_CLDICE_TRAINER.md)
- [Guida Training](docs/START_TRAINING.md)
- [Quick Reference](docs/QUICK_REFERENCE_CLDICE.md)

**🛠️ Tools:**
- [Guida tmux](docs/HOW_TO_USE_TMUX.md)
- [Test Predizioni](docs/GUIDA_TEST_PREDICTIONS.md)
- [Setup Docker](docs/SETUP_ENVIRONMENT.md)

**📝 Storia:**
- [Changelog](docs/CHANGELOG.md)
- [Implementazione clDice](docs/IMPLEMENTAZIONE_CLDICE_COMPLETATA.md)

> ⚠️ **Nota:** Tutta la nuova documentazione .md deve andare in `docs/`!

---

## 🔬 Esperimenti

Il progetto supporta confronti tra diversi setup:

```python
# In test_predictions.py, modifica:
COMPARISON_MODE = 'both'  # 'baseline_only', 'with_cldice_only', 'both'
```

Risultati confronto salvati in `risultati/confronto_both/` con:
- Immagini side-by-side
- Delta metriche (Δ Dice, Δ clDice, etc.)
- Report testuale dettagliato

---

## 🎯 Performance

| Fase | Tempo | GPU | Note |
|------|-------|-----|------|
| Generazione Dataset (2000 img) | ~2-3 ore | No | Dipende da connessione |
| Preprocessing | ~1 min | No | nnU-Net automatic |
| Training 1000 epochs | ~16-20 ore | Sì | ~60 sec/epoch, ~9.8 GB VRAM |
| Inference (400 img) | ~5-10 min | Sì | Batch processing |

---

## 🐛 Troubleshooting

**Training non parte?**
- Verifica variabili ambiente (`echo $nnUNet_raw`)
- Disabilita torch.compile: `nnUNet_compile=False`
- Usa tmux per evitare disconnessioni

**Out of Memory?**
- Riduci batch size in `nnUNetPlans.json`
- Usa GPU con >10 GB VRAM

**clDice loss non funziona?**
- Verifica installazione: il trainer deve essere in `/venv/lib/.../nnunetv2/training/nnUNetTrainer/variants/topology/`
- Controlla logs per errori

Vedi `docs/` per guide dettagliate!

---

## 👤 Autore

**Francesco Girardello**  
PhD Project - Road Segmentation from Satellite Imagery  
Custom nnU-Net Implementation with Topology-Preserving Loss

---

## 📄 Licenza

Proprietaria - PhD Project

---

## 🔗 Link Utili

- [nnU-Net Paper](https://www.nature.com/articles/s41592-020-01008-z)
- [clDice Paper](https://arxiv.org/abs/2003.07311)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [ESRI World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)

---

**Last Updated:** 2025-11-26
