# 🛣️ Road Segmentation from Satellite Imagery using nnU-Net

**PhD Project - Francesco Girardello**

Segmentazione automatica di strade da immagini satellitari ad alta risoluzione usando **nnU-Net**.

---

## 📁 Struttura Progetto

```
/workspace/
├── 📄 made_dataset.py             # Script generazione dataset
├── 🗺️ belgium-roads.osm.pbf       # Dati OSM strade Belgio
│
├── 📁 configs/                    # ⭐ Configurazioni
│   ├── config.yaml                # Config generazione dataset (futuro)
│   └── CONFIGURAZIONE_TRAINING.txt # Config training attuale
│
├── 📁 docs/                       # ⭐ Documentazione
│   ├── README.md                  # Documentazione completa
│   ├── QUICK_START.md             # Guida rapida
│   ├── INSTALL.md                 # Istruzioni installazione
│   └── CHANGELOG.md               # Storia modifiche
│
├── 📁 nnUNet_raw/                 # Dataset per nnU-Net
│   └── Dataset001_Strade/
│       ├── imagesTr/              # Immagini RGB satellitari (2000 img)
│       ├── labelsTr/              # Maschere binarie strade
│       └── dataset.json           # Metadata
│
├── 📁 nnUNet_preprocessed/        # Dati preprocessati
│   └── Dataset001_Strade/
│       ├── nnUNetPlans.json       # Piano training (arch, batch, etc)
│       └── nnUNetPlans_2d/        # Dati processati per training
│
├── 📁 nnUNet_results/             # Risultati training
│   └── Dataset001_Strade/
│       └── nnUNetTrainer__nnUNetPlans__2d/
│           └── fold_0/            # Checkpoint, logs, metriche
│
├── 📁 comparison_results/         # Risultati inference (immagini)
├── 📁 backups/                    # Backup training precedenti
└── 📁 nnUNet/                     # Codebase nnU-Net (modificabile)
```

---

## 🚀 Quick Start

### 1️⃣ **Generazione Dataset**
```bash
python made_dataset.py
# Genera 2000 immagini 512×512 con maschere strade
```

### 2️⃣ **Preprocessing**
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

### 3️⃣ **Training**
```bash
# In tmux:
tmux new -s training
nnUNetv2_train 1 2d 0
# Detach: Ctrl+B poi D
```

### 4️⃣ **Inference**
```bash
nnUNetv2_predict -i input_folder/ -o output_folder/ -d 1 -c 2d -f 0
```

---

## 📚 Documentazione Completa

- **[📖 Documentazione dettagliata](docs/README.md)** - Guida completa
- **[⚡ Quick Start](docs/QUICK_START.md)** - Inizia subito
- **[🛠️ Installazione](docs/INSTALL.md)** - Setup ambiente
- **[📝 Changelog](docs/CHANGELOG.md)** - Storia modifiche

---

## ⚙️ Configurazione

- **[🔧 Configurazioni](configs/)** - File di configurazione
- **Training:** `nnUNet_preprocessed/Dataset001_Strade/nnUNetPlans.json`
- **Dataset:** Modifica parametri in `made_dataset.py` (linee 14-24)

---

## 📊 Dataset

- **2000 immagini** RGB 512×512 px
- **Sorgente immagini:** ESRI World Imagery (zoom 17)
- **Sorgente annotazioni:** OpenStreetMap (Belgium)
- **Classi:** Background (0), Road (1)

---

## 🎯 Performance

| Fase | Tempo | GPU |
|------|-------|-----|
| Generazione Dataset (2000 img) | ~2-3 ore | No |
| Preprocessing | ~1 min | No |
| Training 1000 epoche | ~20 ore | Sì (9.77 GB VRAM) |

---

## 👤 Autore

**Francesco Girardello**  
PhD Project - Road Segmentation from Satellite Imagery

---

## 📄 Licenza

Proprietaria - PhD Project

