import cv2

# Inisialisasi kamera kiri dan kanan
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture('rtsp://admin:0000@192.168.242.70:1935/')

while True:
    retL, frameL = cam_left.read()
    retR, frameR = cam_right.read()
    if not retL or not retR:

        print("Error capturing video")

        break

    # Tampilkan hasil kamera stereo

    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cam_left.release()

cam_right.release()

cv2.destroyAllWindows()