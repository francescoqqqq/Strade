# 🛠️ Setup Environment - Docker/SSH

**Guida per configurare l'ambiente di sviluppo via SSH su Docker**

---

## ✅ Configurazione Completa

L'ambiente è già configurato correttamente! Se stai leggendo questo file, significa che:

- ✅ **Virtual environment** attivo automaticamente al login (`/venv`)
- ✅ **Dipendenze Python** installate (`geopandas`, `nnunetv2`, etc.)
- ✅ **Variabili d'ambiente nnU-Net** configurate in `.bashrc`
- ✅ **Auto-attivazione** del venv ad ogni login SSH

---

## 📋 Cosa è stato configurato

### 1. Variabili d'Ambiente nnU-Net

Le seguenti variabili sono state aggiunte a `~/.bashrc`:

```bash
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
```

### 2. Auto-attivazione Virtual Environment

Il virtual environment (`/venv`) viene attivato automaticamente ad ogni login SSH:

```bash
if [ -d "/venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    source /venv/bin/activate
fi
```

### 3. Dipendenze Installate

Tutte le dipendenze sono installate dal file `/workspace/requirements.txt`:

```
nnunetv2>=2.0
geopandas>=0.14.0
shapely>=2.0.0
pyproj>=3.6.0
fiona>=1.9.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.8.0
numpy>=1.24.0
scipy>=1.11.0
requests>=2.31.0
tqdm>=4.66.0
```

---

## 🧪 Test Ambiente

Per verificare che tutto funzioni correttamente:

```bash
# Test 1: Variabili d'ambiente
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Test 2: Virtual environment
which python
# Dovrebbe mostrare: /venv/bin/python

# Test 3: Dipendenze Python
python -c "import geopandas; import nnunetv2; print('✓ OK')"

# Test 4: GPU (se disponibile)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 🔄 Reinstallare Dipendenze

Se per qualche motivo devi reinstallare le dipendenze:

```bash
# Attiva il venv (se non già attivo)
source /venv/bin/activate

# Installa tutte le dipendenze
pip install -r /workspace/requirements.txt

# Verifica installazione
python -c "import geopandas; print('✓ geopandas OK')"
```

---

## 🆕 Setup Nuovo Container Docker

Se devi configurare un nuovo container Docker da zero:

### Step 1: Installa Python e dipendenze di sistema

```bash
apt-get update
apt-get install -y \
    python3 python3-pip python3-venv \
    libgdal-dev gdal-bin \
    libspatialindex-dev \
    git tmux
```

### Step 2: Crea virtual environment

```bash
python3 -m venv /venv
source /venv/bin/activate
```

### Step 3: Installa dipendenze Python

```bash
pip install --upgrade pip
pip install -r /workspace/requirements.txt
```

### Step 4: Configura .bashrc

```bash
cat >> ~/.bashrc << 'EOF'

# ========================================
# nnU-Net Environment Variables
# ========================================
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"

# Auto-activate Python virtual environment
if [ -d "/venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    source /venv/bin/activate
fi
EOF
```

### Step 5: Ricarica configurazione

```bash
source ~/.bashrc
```

---

## 🐛 Troubleshooting

### Problema: "No module named 'geopandas'"

**Soluzione:**
```bash
source /venv/bin/activate
pip install geopandas shapely pyproj fiona
```

### Problema: Variabili d'ambiente non impostate

**Soluzione:**
```bash
# Verifica che siano in .bashrc
grep "nnUNet" ~/.bashrc

# Se non ci sono, aggiungile:
cat >> ~/.bashrc << 'EOF'
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
EOF

# Ricarica
source ~/.bashrc
```

### Problema: Virtual environment non si attiva automaticamente

**Soluzione:**
```bash
# Aggiungi a .bashrc:
echo 'if [ -d "/venv" ] && [ -z "$VIRTUAL_ENV" ]; then source /venv/bin/activate; fi' >> ~/.bashrc
source ~/.bashrc
```

---

## 📝 Note Importanti

1. **Non usare `.vscode/settings.json`** per le variabili d'ambiente quando lavori via SSH su Docker
   - `.vscode/settings.json` funziona solo per l'IDE locale
   - Usa sempre `~/.bashrc` per configurazioni SSH/Docker

2. **Virtual environment centralizzato** in `/venv`
   - Non serve creare venv separati per progetto
   - Un solo venv condiviso per tutto il container

3. **Dipendenze centralizzate** in `/workspace/requirements.txt`
   - Versione principale per tutto il progetto
   - Anche disponibile in `/workspace/configs/requirements.txt` (legacy)

4. **Persistenza configurazioni**
   - Le modifiche a `~/.bashrc` sono permanenti nel container
   - Le dipendenze Python installate in `/venv` sono permanenti
   - I dati in `/workspace` sono montati da volume esterno

---

## 🎯 Quick Reference

| Cosa | Dove | Come |
|------|------|------|
| **Dipendenze** | `/workspace/requirements.txt` | `pip install -r requirements.txt` |
| **Variabili env** | `~/.bashrc` | `source ~/.bashrc` |
| **Virtual env** | `/venv` | Auto-attivato al login |
| **Test setup** | Terminal | `python made_dataset.py` |

---

**Ultimo aggiornamento:** 20 Novembre 2025  
**Autore:** Francesco Girardello

---

## ✨ Tutto Configurato!

Se hai seguito questi passi, il tuo ambiente è pronto per:
- ✅ Generare dataset con `python made_dataset.py`
- ✅ Preprocessare con `nnUNetv2_plan_and_preprocess -d 1`
- ✅ Trainare con `nnUNetv2_train 1 2d 0`
- ✅ Fare inference con `nnUNetv2_predict`

🚀 **Buon lavoro!**

