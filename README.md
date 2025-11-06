# 🚗 Car Counter – Riconoscimento e Conteggio Auto in Tempo Reale

Questo progetto utilizza **YOLOv8** (un modello di intelligenza artificiale per il computer vision) per riconoscere le auto tramite webcam e mostrare sullo schermo quante sono presenti nel video in quel momento.

---

## ✅ Cosa fa il progetto
- Accende la webcam (o una videocamera esterna)
- Analizza ogni fotogramma usando YOLOv8
- Rileva le auto e le evidenzia con **rettangoli verdi (bounding box)**
- Conta quante auto sono presenti e lo mostra nella finestra video

> In breve: **guardi una strada, il programma riconosce le auto e le conta da solo.**

---

## 🧠 Come funziona (spiegazione semplice)
1. YOLOv8 riceve ogni fotogramma della webcam
2. Il modello individua oggetti nel video (in questo caso solo auto)
3. Il programma disegna il rettangolo attorno all’auto e aggiorna il conteggio
4. Il conteggio viene stabilizzato guardando gli ultimi 1 secondo di rilevazioni per evitare errori di oscillazione

---

## 🔧 Cosa devi installare

### ✅ 1. Installare Python (se non lo hai già)
Versione consigliata: **Python 3.9 o superiore**

---

### ✅ 2. Installare le librerie necessarie

Nella cartella del progetto crea un ambiente virtuale (venv) e scariche le librerie necessarie con questi comandi:

```bash
pip install ultralytics opencv-python
```

Queste librerie servono per:
- `ultralytics` → usare YOLOv8
- `opencv-python` → gestire webcam e immagini

---

### ✅ 3. Scaricare YOLOv8

Non devi scaricare manualmente il modello.  
Al **primo avvio del programma**, YOLOv8 viene scaricato automaticamente:

```python
model = YOLO("yolov8n.pt")
```

Se vuoi un modello più preciso (ma più pesante), puoi usare:
- `yolov8m.pt` (medio)
- `yolov8l.pt` (grande)

Devi solo cambiare il nome nel codice.

---

## ▶️ Come avviare il programma

Esegui questo comando nella cartella del progetto:

```bash
python main.py
```

Se usi una videocamera esterna:

```bash
python main.py --source 1
```

Premi **Q** per chiudere il programma.

---

## 📌 Fine!
Ora puoi usare la webcam per contare auto in tempo reale 😎

Se ti è utile, lascia una ⭐ su GitHub!
