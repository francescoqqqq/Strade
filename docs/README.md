# 🛣️ Road Segmentation from Satellite Imagery using nnU-Net

**PhD Project - Francesco Girardello**

Questo progetto utilizza **nnU-Net** per la segmentazione automatica di strade da immagini satellitari ad alta risoluzione. Il dataset è generato automaticamente da OpenStreetMap (OSM) e tile satellitari ESRI World Imagery.

---

## 📋 Indice

- [Struttura del Progetto](#struttura-del-progetto)
- [Dataset](#dataset)
- [Installazione](#installazione)
- [Generazione Dataset](#generazione-dataset)
- [Training](#training)
- [Inference](#inference)
- [Note Tecniche](#note-tecniche)

---

## 📁 Struttura del Progetto

```
/workspace/
├── made_dataset.py                   # Script principale per generazione dataset
├── belgium-roads.osm.pbf             # Dati OSM strade del Belgio
├── nnUNet_raw/
│   └── Dataset001_Strade/
│       ├── imagesTr/                 # Immagini satellitari RGB (2000 img, 704 MB)
│       ├── labelsTr/                 # Maschere binarie per nnUNet (9.8 MB)
│       ├── labelsTr_viz/             # Maschere visualizzabili 0/255 (debug)
│       ├── allTr/                    # Immagini satellitari + strade sovrapposte (698 MB)
│       └── dataset.json              # Metadata del dataset
├── nnUNet_preprocessed/
│   └── Dataset001_Strade/
│       ├── dataset_fingerprint.json  # Statistiche del dataset
│       ├── nnUNetPlans.json          # Piano di training calcolato
│       └── nnUNetPlans_2d/           # Dati preprocessati per training 2D
└── nnUNet_results/
    └── Dataset001_Strade/
        └── nnUNetTrainer__nnUNetPlans__2d/
            └── fold_0/               # Checkpoint e risultati training
```

---

## 📊 Dataset

### Dataset001_Strade

| Proprietà | Valore |
|-----------|--------|
| **Nome** | Strade |
| **Tipo** | Segmentazione binaria 2D |
| **Immagini** | 2000 (RGB, 512×512 px) |
| **Canali Input** | 3 (R, G, B) |
| **Classi** | 2 (background=0, road=1) |
| **Formato** | PNG |
| **Sorgente Immagini** | ESRI World Imagery (zoom 17) |
| **Sorgente Annotazioni** | OpenStreetMap (Belgium) |
| **Area Coperta** | ~0.005° lat/lon per patch (~500m) |

### Statistiche Intensità (per canale RGB)

| Canale | Min | Max | Mean | Median | Std |
|--------|-----|-----|------|--------|-----|
| **R** | 0 | 255 | 88.6 | 89 | 53.1 |
| **G** | 0 | 255 | 98.4 | 100 | 46.8 |
| **B** | 0 | 255 | 79.3 | 78 | 45.8 |

### Configurazione nnU-Net

```json
{
  "architecture": "2D U-Net (8 stages)",
  "patch_size": [512, 512],
  "batch_size": 12,
  "normalization": "ZScoreNormalization (per canale RGB)",
  "spacing": [1.0, 1.0],
  "features_per_stage": [32, 64, 128, 256, 512, 512, 512, 512]
}
```

---

## 🛠️ Installazione

### Prerequisiti

```bash
# Python 3.10+
# CUDA (per GPU training)
```

### Dipendenze

```bash
pip install nnunetv2
pip install geopandas shapely pillow requests matplotlib numpy
```

### Setup Variabili d'Ambiente

```bash
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
```

---

## 🚀 Generazione Dataset

### Script: `made_dataset.py`

Lo script `made_dataset.py` genera automaticamente il dataset combinando:
1. **Tile satellitari** da ESRI World Imagery
2. **Geometrie strade** da file OSM (`.pbf`)

### Configurazione

Modifica le seguenti variabili in `made_dataset.py`:

```python
# === CONFIGURAZIONE ===
osm_file = "/workspace/belgium-roads.osm.pbf"  # File OSM
dataset_id = "001"                              # ID dataset nnUNet
dataset_name = "Strade"                         # Nome dataset
num_images = 2000                               # Numero immagini da generare
image_size = 512                                # Dimensione immagine (px)
max_attempts = 100                              # Tentativi per trovare patch con strade
data = "imm,lab,all"                           # Opzioni output (imm/lab/all)
```

### Opzioni Output (`data`)

| Opzione | Descrizione | Cartella |
|---------|-------------|----------|
| `imm` | Immagini satellitari RGB | `imagesTr/` |
| `lab` | Maschere binarie strade (0/1) | `labelsTr/` |
| `all` | Immagini satellitari + strade sovrapposte | `allTr/` |

**Esempio:**
- `data = "imm,lab"` → Genera solo immagini e maschere per training
- `data = "all"` → Genera solo visualizzazioni composite
- `data = "imm,lab,all"` → Genera tutto

### Esecuzione

```bash
python made_dataset.py
```

**Output atteso:**
```
Caricamento dati OSM...
✓ Caricate 123456 strade
Conversione coordinate a WGS84...
✓ Coordinate convertite

Generazione 2000 immagini con strade...

Patch 1/2000 - Centro: (4.3521, 50.8503)
  Trovate 15 strade
  Scaricando immagine satellitare...
  Processando immagine satellitare...
  Creando immagine satellitare + strade...
  Creando maschera strade binaria...
  ✓ Salvata

...

✓ COMPLETATO!
```

---

## 🎓 Training

### 1. Planning e Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Questo comando:
- ✅ Verifica l'integrità del dataset
- 📊 Analizza le statistiche delle immagini
- 📐 Calcola la configurazione ottimale (patch size, batch size, architettura)
- 🔄 Preprocessa tutte le immagini (normalizzazione, cropping)

**Output:**
- `nnUNet_preprocessed/Dataset001_Strade/nnUNetPlans.json`
- `nnUNet_preprocessed/Dataset001_Strade/dataset_fingerprint.json`

### 2. Training

```bash
# Training completo (5-fold cross-validation)
nnUNetv2_train 1 2d all

# Training singolo fold (più veloce per test)
nnUNetv2_train 1 2d 0

# Training con GPU specifica
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 2d 0
```

**Parametri:**
- `1` → Dataset ID (Dataset001)
- `2d` → Configurazione (2D U-Net)
- `0` / `all` → Fold (0-4 o tutti)

### 3. Monitoraggio Training

I checkpoint e log vengono salvati in:
```
nnUNet_results/Dataset001_Strade/nnUNetTrainer__nnUNetPlans__2d/fold_X/
```

---

## 🔮 Inference

### Predizione su Nuove Immagini

```bash
nnUNetv2_predict \
  -i /path/to/input/images \
  -o /path/to/output/predictions \
  -d 1 \
  -c 2d \
  -f 0
```

**Parametri:**
- `-i` → Cartella immagini input (formato: `*.png` o `*_0000.png`)
- `-o` → Cartella output predizioni
- `-d` → Dataset ID (1)
- `-c` → Configurazione (2d)
- `-f` → Fold (0-4 o ometti per ensemble)

### Ensemble di Tutti i Fold

```bash
# Ometti il parametro -f per usare tutti i fold
nnUNetv2_predict \
  -i /path/to/input/images \
  -o /path/to/output/predictions \
  -d 1 \
  -c 2d
```

---

## 📝 Note Tecniche

### Formato Immagini

**Input (imagesTr/):**
- Nome: `strade_XXXX_0000.png`
- Formato: PNG RGB (3 canali)
- Dimensione: 512×512 pixels
- Suffisso `_0000` richiesto da nnUNet per indicare il canale 0

**Label (labelsTr/):**
- Nome: `strade_XXXX.png`
- Formato: PNG Grayscale (1 canale)
- Valori: 0 (background), 1 (road)
- ⚠️ **Importante:** I valori sono 0/1, non 0/255!

**Label Visualizzazione (labelsTr_viz/):**
- Come `labelsTr/` ma con valori 0/255 per visualizzazione
- Non usate per training, solo per debug/ispezione

### Coordinate e Proiezioni

- **Input OSM:** EPSG:4326 (WGS84)
- **Tile Satellitari:** Web Mercator (OSM tiles standard)
- **Zoom Level:** 17 (~2.39m/pixel @ equatore)
- **Patch Size:** ~0.005° lat/lon (~500m lato)

### Algoritmo di Campionamento

Lo script `made_dataset.py` usa un algoritmo di campionamento casuale con controllo:

1. **Campionamento casuale** di coordinate entro i bounds OSM
2. **Verifica presenza strade** nella patch
3. **Scaricamento tile 3×3** centrati sulla patch
4. **Cropping e resize** alla dimensione target (512×512)
5. **Rasterizzazione strade** su maschera binaria

⚠️ **Fallback:** Se dopo `max_attempts` tentativi non trova patch con strade, lo script continua ma segnala warning.

### Performance e Risorse

| Fase | Tempo (2000 img) | RAM | GPU |
|------|------------------|-----|-----|
| **Generazione Dataset** | ~2-3 ore | 8 GB | No |
| **Preprocessing** | ~1 minuto | 16 GB | No |
| **Training (1 fold)** | ~2-4 ore | 16 GB | Sì (consigliato) |

**GPU consigliata:** NVIDIA con ≥8GB VRAM

---

## 🐛 Troubleshooting

### Errore: `bash: !': event not found`

Quando usi comandi Python con `!` nelle stringhe:

```bash
# ❌ NON FUNZIONA
python -c "print('Test OK!')"

# ✅ SOLUZIONE: Inverti le virgolette
python -c 'print("Test OK!")'

# ✅ ALTERNATIVA: Escape del punto esclamativo
python -c "print('Test OK\!')"
```

### Errore: NaturalImage2DIO RGB mismatch

Se nnUNet si lamenta del numero di canali:
- Verifica che le immagini siano RGB (3 canali)
- Verifica il suffisso `_0000` nel nome file
- Controlla `channel_names` in `dataset.json`

### Maschere non visualizzate correttamente

Le maschere in `labelsTr/` hanno valori 0/1 (non visibili a occhio).
Usa le maschere in `labelsTr_viz/` con valori 0/255 per visualizzazione.

---

## 📚 Riferimenti

- **nnU-Net:** [GitHub](https://github.com/MIC-DKFZ/nnUNet) | [Paper](https://arxiv.org/abs/1904.08128)
- **OpenStreetMap:** [https://www.openstreetmap.org](https://www.openstreetmap.org)
- **ESRI World Imagery:** [ArcGIS Online](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)

---

## 📄 Licenza

**Proprietaria** - Francesco Girardello PhD Project

---

## 👤 Autore

**Francesco Girardello**  
PhD Project - Road Segmentation from Satellite Imagery

---

## 🎯 TODO / Future Work

- [ ] Aggiungere validazione dataset (split train/val)
- [ ] Implementare data augmentation custom
- [ ] Testare configurazioni 3D/3D fullres
- [ ] Estendere a multiple regioni geografiche
- [ ] Post-processing per rimozione rumore
- [ ] Metriche custom per topologia strade
- [ ] Export modello per deployment

---

**Ultimo aggiornamento:** Novembre 2025

