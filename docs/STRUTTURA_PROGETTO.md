# 📁 Struttura del Progetto

Questa guida descrive l'organizzazione delle cartelle e dei file del progetto.

---

## 🗂️ Root Directory

```
workspace/
├── 📄 made_dataset.py             # Script principale per generazione dataset
├── 📄 README.md                   # Documentazione panoramica (punta a docs/)
├── 📄 requirements.txt            # Dipendenze Python
├── 📄 check_gpu.py                # Utility verifica GPU
├── 📄 test_predictions.py         # Script test inference
├── 📄 visualize_samples.py        # Script visualizzazione dataset
│
├── 🗺️ belgium-roads.osm.pbf       # Dati OpenStreetMap strade Belgio (1.2 GB)
│
├── 📁 configs/                    # ⭐ CONFIGURAZIONI
├── 📁 docs/                       # ⭐ DOCUMENTAZIONE
├── 📁 nnUNet_raw/                 # Dataset nnU-Net
├── 📁 nnUNet_preprocessed/        # Dati preprocessati
├── 📁 nnUNet_results/             # Risultati training
├── 📁 comparison_results/         # Output inference
├── 📁 backups/                    # Backup training precedenti
└── 📁 nnUNet/                     # Codebase nnU-Net (modificabile)
```

---

## 📁 Cartelle Principali

### **`configs/`** - Configurazioni

File di configurazione per dataset generation e training:

```
configs/
├── config.yaml                    # Config per made_dataset.py (futuro)
├── CONFIGURAZIONE_TRAINING.txt    # Riepilogo config training attuale
└── README.md                      # Documentazione configurazioni
```

**Uso:**
- `config.yaml`: Parametri generazione dataset (da implementare in made_dataset.py)
- `CONFIGURAZIONE_TRAINING.txt`: Info configurazione training corrente

---

### **`docs/`** - Documentazione

Tutta la documentazione del progetto:

```
docs/
├── README.md                      # Guida completa e dettagliata
├── QUICK_START.md                 # Guida rapida per iniziare
├── INSTALL.md                     # Istruzioni installazione
├── CHANGELOG.md                   # Storia modifiche e versioni
└── STRUTTURA_PROGETTO.md          # Questo file
```

**Uso:**
- Leggi `README.md` per documentazione completa
- Usa `QUICK_START.md` per partire subito
- Consulta `INSTALL.md` per setup ambiente

---

### **`nnUNet_raw/`** - Dataset Originale

Dataset grezzo per nnU-Net (formato nnU-Net):

```
nnUNet_raw/
└── Dataset001_Strade/
    ├── imagesTr/                  # 2000 immagini RGB 512×512 (704 MB)
    │   └── strade_XXXX_0000.png
    ├── labelsTr/                  # Maschere binarie 0/1 (9.8 MB)
    │   └── strade_XXXX.png
    ├── labelsTr_viz/              # Maschere visualizzabili 0/255 (debug)
    │   └── strade_XXXX.png
    ├── allTr/                     # Satellitare + strade sovrapposte (698 MB)
    │   └── strade_XXXX.png
    └── dataset.json               # Metadata dataset
```

**Note:**
- Le immagini in `labelsTr/` hanno valori **0/1** (per nnU-Net)
- Le immagini in `labelsTr_viz/` hanno valori **0/255** (per visualizzazione)
- Non usare `labelsTr_viz/` per training!

---

### **`nnUNet_preprocessed/`** - Dati Preprocessati

Dati processati da nnU-Net (normalizzati, cropped):

```
nnUNet_preprocessed/
└── Dataset001_Strade/
    ├── nnUNetPlans.json           # ⭐ Piano training (architettura, batch, etc)
    ├── dataset_fingerprint.json   # Statistiche dataset
    ├── splits_final.json          # Split train/val
    ├── gt_segmentations/          # Ground truth per validation
    └── nnUNetPlans_2d/            # Dati preprocessati (npz files)
```

**File importante:**
- **`nnUNetPlans.json`**: Contiene configurazione rete (stages, batch size, features)

---

### **`nnUNet_results/`** - Risultati Training

Checkpoint, log e metriche del training:

```
nnUNet_results/
└── Dataset001_Strade/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_final.pth        # Checkpoint finale
            ├── checkpoint_best.pth         # Best checkpoint (val dice)
            ├── checkpoint_latest.pth       # Ultimo checkpoint (resume)
            ├── progress.png                # Grafico loss/dice
            ├── training_log_*.txt          # Log training
            └── validation_raw/             # Output validazione
```

**Note:**
- Usa `checkpoint_best.pth` per inference
- Usa `checkpoint_latest.pth` per continuare training (`--c`)

---

### **`comparison_results/`** - Output Inference

Risultati inference con comparazioni:

```
comparison_results/
└── strade_XXXX_comparison.png     # Griglia: input | GT | prediction
```

---

### **`backups/`** - Backup

Backup di training precedenti o configurazioni:

```
backups/
└── Dataset001_Strade_training_old_YYYYMMDD/
```

---

### **`nnUNet/`** - Codebase nnU-Net

Repository nnU-Net completo (per sviluppo/debug):

```
nnUNet/
├── nnunetv2/                      # Package principale
│   ├── training/                  # Training logic
│   ├── inference/                 # Inference logic
│   └── ...
└── documentation/                 # Doc nnU-Net originale
```

**Nota:** Di solito non serve modificare, ma utile per debug o custom trainer.

---

## 🔍 File Specifici

### **`made_dataset.py`**
Script principale per generazione dataset:
- Scarica tile satellitari (ESRI World Imagery)
- Estrae geometrie strade da OSM
- Genera immagini RGB + maschere binarie
- Configurazione: linee 14-24

### **`config.yaml`**
Configurazione futura per `made_dataset.py` (non ancora implementato):
- Dataset ID e nome
- File OSM input
- Parametri generazione (num_images, size, etc)
- Filtri strade OSM
- Server tile satellitari

### **`nnUNetPlans.json`**
Configurazione training nnU-Net (autogenerato da preprocessing):
- Architettura rete (stages, features, strides)
- Batch size
- Patch size
- Normalizzazione

### **`.gitignore`**
Esclude da git:
- File grandi (immagini PNG, OSM PBF)
- Checkpoint training
- Cache Python
- File temporanei

---

## 📊 Dimensioni Tipiche

| Cartella | Dimensione | Note |
|----------|------------|------|
| `nnUNet_raw/imagesTr/` | ~700 MB | 2000 immagini RGB |
| `nnUNet_raw/labelsTr/` | ~10 MB | Maschere binarie |
| `nnUNet_preprocessed/` | ~800 MB | Dati processati |
| `nnUNet_results/fold_0/` | ~200-500 MB | Checkpoint training |
| `belgium-roads.osm.pbf` | ~1.2 GB | Dati OSM Belgio |
| **TOTALE** | **~3-4 GB** | Senza backup |

---

## 🚮 Pulizia

### Liberare spazio (se necessario):

```bash
# Rimuovi immagini intermediate (allTr, labelsTr_viz)
rm -rf nnUNet_raw/Dataset001_Strade/allTr/
rm -rf nnUNet_raw/Dataset001_Strade/labelsTr_viz/

# Rimuovi backup vecchi
rm -rf backups/

# Rimuovi risultati inference
rm -rf comparison_results/*.png

# ⚠️ NON rimuovere:
# - nnUNet_raw/Dataset001_Strade/imagesTr/  (necessario)
# - nnUNet_raw/Dataset001_Strade/labelsTr/  (necessario)
# - nnUNet_preprocessed/                    (necessario)
# - nnUNet_results/                         (checkpoint!)
```

---

## 🔄 Workflow Tipico

1. **Generazione Dataset:**
   ```bash
   python made_dataset.py
   # Output → nnUNet_raw/Dataset001_Strade/
   ```

2. **Preprocessing:**
   ```bash
   nnUNetv2_plan_and_preprocess -d 1
   # Output → nnUNet_preprocessed/Dataset001_Strade/
   ```

3. **Training:**
   ```bash
   nnUNetv2_train 1 2d 0
   # Output → nnUNet_results/Dataset001_Strade/.../fold_0/
   ```

4. **Inference:**
   ```bash
   nnUNetv2_predict -i input/ -o output/ -d 1 -c 2d -f 0
   # Usa checkpoint da nnUNet_results/.../checkpoint_best.pth
   ```

---

**Ultimo aggiornamento:** Novembre 2025

