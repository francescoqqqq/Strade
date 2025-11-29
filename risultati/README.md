# 📊 Confronto Esperimenti: Baseline vs clDice Loss

Questa directory contiene i risultati degli esperimenti per la segmentazione delle strade.

## 📁 Struttura

```
risultati/
├── baseline_dice_only/          ← Modello trainato solo con Dice Loss (già completato)
│   ├── Confronto_imm/           → 20 immagini di test con predizioni
│   └── risultati_test.txt       → Report con metriche: Dice, IoU, Accuracy, clDice
│
└── with_cldice_loss/            ← Modello da trainare con Dice + clDice Loss (futuro)
    ├── Confronto_imm/           → Stesse 20 immagini per confronto diretto
    └── risultati_test.txt       → Report con metriche
```

## 🎯 Obiettivo

Confrontare le performance tra:
- **Baseline**: modello trainato con standard Dice Loss
- **clDice**: modello trainato con Dice Loss + clDice Loss (per migliorare topologia)

## 📈 Metriche Valutate

- **Dice Coefficient**: overlap generale
- **IoU**: intersection over union
- **Pixel Accuracy**: accuratezza pixel-level
- **clDice**: preservazione della topologia (centerline)

## 🔄 Workflow

1. ✅ Baseline trainato e testato
2. ⏳ Implementare clDice nella loss function
3. ⏳ Trainare nuovo modello
4. ⏳ Testare sulle stesse immagini
5. ⏳ Confrontare risultati side-by-side

---
*Generato: 2025-11-25*

