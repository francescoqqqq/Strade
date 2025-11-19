# 📁 Configurazioni

Questa cartella contiene i file di configurazione per il progetto.

---

## 📄 File presenti:

### **`config.yaml`**
Configurazione per la generazione del dataset con `made_dataset.py`.

**Uso futuro (da implementare):**
```bash
python made_dataset.py --config configs/config.yaml
```

**Parametri configurabili:**
- **Dataset:** ID, nome, descrizione
- **Input:** File OSM, regione geografica
- **Generazione:** Numero immagini, dimensioni, patch size
- **Satellite:** Server tiles, zoom level
- **Roads:** Larghezza linee, tipi highway OSM
- **Output:** Directory base, opzioni visualizzazione
- **nnU-Net:** Directory raw/preprocessed/results

---

### **`CONFIGURAZIONE_TRAINING.txt`**
Riepilogo della configurazione attuale per il training nnU-Net.

**Contiene:**
- Architettura rete (stages, features, batch size)
- Parametri training
- Tempi attesi
- Comandi per avviare il training

---

## 🎯 Note:

- **`config.yaml` NON è ancora supportato da `made_dataset.py`** (per ora i parametri sono hardcoded nel file .py)
- Se vuoi modificare parametri di generazione dataset, edita direttamente `made_dataset.py` (linee 14-24)
- Per configurazione training nnU-Net, vedi `nnUNet_preprocessed/Dataset001_Strade/nnUNetPlans.json`

---

## 🔮 Implementazione futura `config.yaml`:

Se vuoi che `made_dataset.py` legga da `config.yaml`, possiamo implementare:
```python
import yaml

# In made_dataset.py, aggiungi:
if len(sys.argv) > 2 and sys.argv[1] == '--config':
    with open(sys.argv[2], 'r') as f:
        config = yaml.safe_load(f)
    osm_file = config['input']['osm_file']
    dataset_id = config['dataset']['id']
    # ... etc
```

Fammi sapere se vuoi che lo implementi! 🚀

