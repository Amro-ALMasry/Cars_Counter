import cv2
from ultralytics import YOLO
import time
from collections import deque

# Carica modello YOLO (rileva auto)
model = YOLO("yolov8n.pt") #puioi cambiare con yolov8m.pt o yolov8l.pt per modelli più grandi

# Sorgente della camera (0 = webcam, cambia se necessario)
cap = cv2.VideoCapture(3)

SAMPLE_DURATION = 1.0   # durata finestra campionamento (secondi)
FPS_TARGET = 25         # limite FPS
conteggi_buffer = deque()
ultimo_aggiornamento = time.time()
conteggio_stabile = 0

ultimo_frame_time = time.time()
frame_interval = 1.0 / FPS_TARGET

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Errore: frame non valido dalla camera.")
        continue

    tempo_corrente = time.time()

    # Mantieni FPS costante
    if tempo_corrente - ultimo_frame_time < frame_interval:
        cv2.imshow("Contatore Auto", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    ultimo_frame_time = tempo_corrente

    # Rilevamento: class 2 = "car", class 7 = "truck", class 5 = "bus"
    results = model(frame, classes=[2], verbose=False, conf=0.6)

    # Conta le auto nel frame attuale
    num_auto_corrente = len(results[0].boxes)

    # Salva il conteggio nel buffer
    conteggi_buffer.append({
        "count": num_auto_corrente,
        "time": tempo_corrente
    })

    # Mantieni solo i conteggi nell'ultimo secondo
    while conteggi_buffer and (tempo_corrente - conteggi_buffer[0]["time"]) > SAMPLE_DURATION:
        conteggi_buffer.popleft()

    # Conteggio stabile (massimo degli ultimi 1s)
    if conteggi_buffer:
        conteggio_stabile = max(item["count"] for item in conteggi_buffer)

    # Disegna box sulle auto attuali
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostra testo del conteggio
    cv2.putText(frame, f"Auto presenti: {conteggio_stabile}", (50, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3)

    # Conteggio corrente (debug)
    cv2.putText(frame, f"Rilevate: {num_auto_corrente}", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Contatore Auto", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
