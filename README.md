# 🚗 Car Counter – Riconoscimento e Conteggio Auto in Tempo Reale

Questo progetto utilizza **YOLOv8** per riconoscere e contare automaticamente le auto in tempo reale tramite webcam.

> In breve: il programma analizza il video della webcam, rileva le auto e mostra quante sono presenti sullo schermo.

---

## ✅ Cosa fa il progetto

- Avvia la webcam (o una videocamera esterna)
- Analizza ogni frame del video
- Rileva le auto usando YOLOv8
- Disegna un rettangolo verde attorno a ogni auto
- Mostra il numero di auto presenti in tempo reale
- Stabilizza il conteggio per evitare oscillazioni (usa gli ultimi 1 secondo di rilevazioni)

---

## 📦 Librerie utilizzate

Il progetto utilizza le seguenti librerie principali:

- **OpenCV** ```pip3 install opencv-python```  
- **Ultralytics** ```pip3 install ultralytics```  
- **PyTorch, torchvision, torchaudio** ```pip3 install torch torchvision torchaudio```
---

## 📚 Dipendenze aggiuntive

Queste vengono installate automaticamente insieme a Ultralytics:

- `numpy` → gestione dati numerici  
- `pillow` → gestione immagini  
- `torchvision` → supporto per PyTorch  
- `torchaudio` → (incluso con PyTorch, non sempre usato)
---

## 🐍 Librerie standard (già incluse in Python)

Non serve installarle:

- `time`
- `collections`
---
