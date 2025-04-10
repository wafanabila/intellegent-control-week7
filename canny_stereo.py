import cv2

import numpy as np



def canny_edge_detection(image):

    """Fungsi untuk mendeteksi tepi menggunakan Canny"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    return edges



# Capture stereo images

cam_left = cv2.VideoCapture(0)

cam_right = cv2.VideoCapture('rtsp://admin:0000@192.168.242.70:1935/')



while True:

    retL, frameL = cam_left.read()

    retR, frameR = cam_right.read()



    if not retL or not retR:

        break



    # Proses Canny Edge Detection

    edgesL = canny_edge_detection(frameL)

    edgesR = canny_edge_detection(frameR)



    # Tampilkan hasil

    cv2.imshow("Canny Left", edgesL)

    cv2.imshow("Canny Right", edgesR)



    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cam_left.release()

cam_right.release()

cv2.destroyAllWindows()