import cv2
import numpy as np
import csv
from datetime import datetime

# --- Konfigurasi Asumsi Kamera ---
focal_length = 700  # pixel
baseline = 6.0      # cm (jarak antar kamera, ukur manual!)

# --- Inisialisasi Kamera ---
capL = cv2.VideoCapture(1)  # Ganti jika perlu
capR = cv2.VideoCapture(2)

# Menetapkan resolusi dan mengecek jika kamera berhasil dibuka
if not capL.isOpened() or not capR.isOpened():
    print("❌ Kamera gagal dibuka. Pastikan kamera terhubung dengan benar.")
    exit()

capL.set(3, 640)
capL.set(4, 480)
capR.set(3, 640)
capR.set(4, 480)

# --- Callback untuk klik mouse ---
depth_data = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity = disp[y, x]
        if disparity > 0:
            depth_cm = (focal_length * baseline) / disparity
            print(f"[📏] Titik ({x},{y}) -> Depth: {depth_cm:.2f} cm")
            depth_data.append((x, y, depth_cm))

# --- Ambil Gambar Stereo Sekaligus ---
print("📸 Mengambil gambar stereo...")
retL, frameL = capL.read()
retR, frameR = capR.read()

if not retL or not retR:
    print("❌ Gagal mengambil gambar dari kamera.")
    capL.release()
    capR.release()
    exit()

cv2.imwrite("left.jpg", frameL)
cv2.imwrite("right.jpg", frameR)
print("✅ Gambar disimpan: left.jpg & right.jpg")

# --- Konversi ke grayscale ---
grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

# --- Buat stereo matcher ---
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# --- Normalisasi untuk ditampilkan ---
disp_display = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
disp_display = np.uint8(disp_display)

# --- Tampilkan & ambil titik ---
cv2.namedWindow("Stereo Disparity")
cv2.setMouseCallback("Stereo Disparity", mouse_callback)

print("🖱 Klik di titik yang ingin diukur (tekan 'q' untuk keluar)...")

while True:
    cv2.imshow("Stereo Disparity", disp_display)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# --- Simpan ke CSV ---
filename = f"rail_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y", "depth_cm"])
    writer.writerows(depth_data)

print(f"💾 Data kedalaman disimpan ke: {filename}")

# --- Bersih-bersih ---
capL.release()
capR.release()
cv2.destroyAllWindows()
