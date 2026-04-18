import cv2
from ultralytics import YOLO
import time
from collections import deque
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(1)

CONF = 0.6

# ZONA parcheggio (evita strada): regola questi
Y_MIN = 200
Y_MAX = 900

# Raggruppamento in righe
ROW_Y_TOL_FACTOR = 0.60

# ====== CALIBRAZIONE (PIXEL -> METRI) ======
# Valore da tarare. Esempio: 0.01 significa 1 pixel = 1 cm.
METERS_PER_PIXEL = 0.012  # <-- da tarare

# ====== QUANTE AUTO CI STANNO NEL GAP (in metri) ======
CAR_SLOT_METERS = 4.6     # lunghezza media auto ~4.3-4.7m (scegli tu)
MARGIN_METERS = 0.6       # spazio extra tra auto (scegli tu)

FPS_TARGET = 25
frame_interval = 1.0 / FPS_TARGET
ultimo_frame_time = time.time()

SAMPLE_DURATION = 1.0
free_buffer = deque()
free_stable = 0

def bottom_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int(y2)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    now = time.time()
    if now - ultimo_frame_time < frame_interval:
        cv2.imshow("Parcheggio (metri)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    ultimo_frame_time = now

    results = model(frame, classes=[2], conf=CONF, verbose=False)

    cars = []
    widths, heights = [], []

    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx, cy = bottom_center(x1, y1, x2, y2)

        # filtro parcheggio (per non prendere auto strada)
        if not (Y_MIN <= cy <= Y_MAX):
            continue

        w = x2 - x1
        h = y2 - y1
        widths.append(w)
        heights.append(h)

        cars.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy, "w": w, "h": h})

    # linee debug zona parcheggio
    cv2.line(frame, (0, Y_MIN), (frame.shape[1], Y_MIN), (255, 255, 0), 2)
    cv2.line(frame, (0, Y_MAX), (frame.shape[1], Y_MAX), (255, 255, 0), 2)

    free_slots_est = 0

    if len(cars) >= 2:
        avg_h = sum(heights) / len(heights)
        ROW_Y_TOL = avg_h * ROW_Y_TOL_FACTOR

        # crea righe (cluster su cy)
        rows = []
        for car in sorted(cars, key=lambda c: c["cy"]):
            placed = False
            for row in rows:
                if abs(car["cy"] - row["mean_cy"]) <= ROW_Y_TOL:
                    row["cars"].append(car)
                    row["mean_cy"] = sum(c["cy"] for c in row["cars"]) / len(row["cars"])
                    placed = True
                    break
            if not placed:
                rows.append({"mean_cy": car["cy"], "cars": [car]})

        for r_i, row in enumerate(rows, start=1):
            row_cars = row["cars"]
            if len(row_cars) < 2:
                continue

            # ordina in orizzontale
            row_cars.sort(key=lambda c: c["x1"])

            # scritta riga
            cv2.putText(frame, f"Riga {r_i}", (10, int(row["mean_cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for i in range(len(row_cars) - 1):
                c1 = row_cars[i]
                c2 = row_cars[i + 1]

                # gap in pixel tra bordi box
                gap_px = c2["x1"] - c1["x2"]
                if gap_px <= 0:
                    continue

                gap_m = gap_px * METERS_PER_PIXEL

                # mostra sempre la distanza tra la coppia (in metri)
                y_line = int((c1["cy"] + c2["cy"]) / 2)
                cv2.line(frame, (c1["x2"], y_line), (c2["x1"], y_line), (0, 0, 255), 2)

                midx = int((c1["x2"] + c2["x1"]) / 2)
                cv2.putText(frame, f"{gap_m:.2f} m", (midx - 30, y_line - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # quante auto ci stanno nello spazio?
                slot_unit = CAR_SLOT_METERS + MARGIN_METERS
                slots_here = int(gap_m // slot_unit)

                if slots_here > 0:
                    free_slots_est += slots_here
                    cv2.putText(frame, f"+{slots_here}", (midx - 10, y_line + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # stabilizzazione 1s
    free_buffer.append({"free": free_slots_est, "time": now})
    while free_buffer and (now - free_buffer[0]["time"]) > SAMPLE_DURATION:
        free_buffer.popleft()
    free_stable = max((x["free"] for x in free_buffer), default=0)

    # disegna box
    for c in cars:
        cv2.rectangle(frame, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)

    cv2.putText(frame, f"Auto (parcheggio): {len(cars)}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Posti liberi stimati: {free_stable}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Parcheggio (metri)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
