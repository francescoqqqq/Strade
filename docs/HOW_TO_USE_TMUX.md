# Come Usare tmux per Training nnU-Net

## Installazione (se necessario)
```bash
apt-get update && apt-get install -y tmux
```

## Avviare Training in tmux

### 1. Creare una nuova sessione tmux
```bash
tmux new -s nnunet_training
```

### 2. Dentro tmux, avviare il training
```bash
cd /workspace
source /venv/bin/activate
nnUNet_compile=False nnUNetv2_train 001 2d 0 -tr nnUNetTrainerClDice
```

### 3. Distaccarsi dalla sessione (lasciare il training in esecuzione)
Premi: `Ctrl+B` poi `D`

Ora puoi chiudere il terminale, il training continuerà!

## Riattaccarsi alla Sessione

### Quando ti riconnetti:
```bash
# Vedere le sessioni tmux attive
tmux ls

# Riattaccarsi alla sessione
tmux attach -t nnunet_training
```

Vedrai l'output del training in tempo reale!

## Comandi Utili tmux

- **Distaccarsi**: `Ctrl+B` poi `D`
- **Killare una sessione**: `tmux kill-session -t nnunet_training`
- **Creare una nuova finestra**: `Ctrl+B` poi `C`
- **Switchare tra finestre**: `Ctrl+B` poi numero (0, 1, 2...)
- **Scrollare l'output**: `Ctrl+B` poi `[` (poi frecce su/giù, `Q` per uscire)

## Vantaggi di tmux

✅ Puoi riattaccarti e vedere l'output live
✅ Puoi avere multiple finestre (training, monitoring, editing)
✅ Più robusto di nohup per sessioni lunghe
✅ Puoi scrollare indietro nell'output

## Tip: Monitoraggio

Puoi aprire 2 finestre tmux:
1. Finestra 1: Training
2. Finestra 2: Monitoring con `watch -n 10 nvidia-smi` e `tail -f training.log`
