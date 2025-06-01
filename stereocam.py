import cv2
import numpy as np
import csv
from datetime import datetime

# ======================
# PARAMETER KAMERA STEREO
# ======================
FOCAL_LENGTH = 700  # dalam pixel (hasil kalibrasi atau estimasi)
BASELINE = 6.0      # dalam cm (jarak antar kamera)

# ======================
# FUNGSI BANTUAN
# ======================
def save_to_csv(data, filename="geometry_output.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def get_distance_cm(disparity_value):
    if disparity_value <= 0:
        return None
    return (FOCAL_LENGTH * BASELINE) / disparity_value

# Header CSV
save_to_csv(["Timestamp", "Rail_Width_cm", "Rail_Length_cm", "Slope_deg", "Ballast_Deformation"])

# ======================
# INISIALISASI KAMERA
# ======================
cap_left = cv2.VideoCapture(1)
cap_right = cv2.VideoCapture(2)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Gagal membaca kamera.")
        break

    frame_left = cv2.resize(frame_left, (640, 480))
    frame_right = cv2.resize(frame_right, (640, 480))

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right)
    disparity_float = disparity.astype(np.float32) / 16.0
    disp_norm = cv2.normalize(disparity_float, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # ==========================
    # EDGE DETECTION + KONTOUR
    # ==========================
    edges = cv2.Canny(gray_left, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rail_width_cm = 0
    rail_length_cm = 0
    slope_deg = 0
    ballast_deformation = 0

    contour_centers = []
    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                contour_centers.append((cx, cy, w, h))
                cv2.rectangle(frame_left, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(contour_centers) >= 2:
            contour_centers = sorted(contour_centers, key=lambda c: c[0])
            cx1, cy1, w1, h1 = contour_centers[0]
            cx2, cy2, w2, h2 = contour_centers[-1]

            rail_width_px = abs(cx2 - cx1)
            slope_rad = np.arctan2(cy2 - cy1, cx2 - cx1)
            slope_deg = np.degrees(slope_rad)

            # Gunakan rata-rata posisi untuk ambil disparity
            cx_mid = (cx1 + cx2) // 2
            cy_mid = (cy1 + cy2) // 2
            patch = disparity_float[cy_mid-2:cy_mid+2, cx_mid-2:cx_mid+2]
            valid = patch[patch > 0]
            if len(valid) > 0:
                disp_val = np.mean(valid)
                distance_cm = get_distance_cm(disp_val)
                if distance_cm:
                    rail_width_cm = (rail_width_px * distance_cm) / FOCAL_LENGTH

            # ==========================
            # PANJANG REL: berdasarkan tinggi kontur
            # ==========================
            rail_length_px = max(h1, h2)
            y_mid = (cy1 + cy2) // 2
            patch = disparity_float[y_mid-2:y_mid+2, cx_mid-2:cx_mid+2]
            valid = patch[patch > 0]
            if len(valid) > 0:
                disp_val = np.mean(valid)
                distance_cm = get_distance_cm(disp_val)
                if distance_cm:
                    rail_length_cm = (rail_length_px * distance_cm) / FOCAL_LENGTH

    # ==========================
    # DEFORMASI BALLAST
    # ==========================
    roi = disparity_float[300:400, :]
    valid = roi[roi > 0]
    if len(valid) > 0:
        ballast_deformation = round(np.std(valid), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_to_csv([timestamp, round(rail_width_cm, 2), round(rail_length_cm, 2), round(slope_deg, 2), ballast_deformation])

    cv2.putText(frame_left, f"Width: {round(rail_width_cm,2)} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame_left, f"Length: {round(rail_length_cm,2)} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame_left, f"Slope: {round(slope_deg,2)} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame_left, f"Deformation: {ballast_deformation}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Disparity Map", disp_norm)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()