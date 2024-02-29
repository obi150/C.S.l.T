import cv2
import urllib.request
import numpy as np

cap = cv2.VideoCapture('http://your_ip_address:81/stream')

while True:
    if cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("ESP32-cam_Capture", frame)
        cv2.waitKey(1)

cv2.destroyAllWindows()
