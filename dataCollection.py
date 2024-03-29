import cv2
import mediapipe as mp
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

offset = 1.5
imgSize = 300

_, img = cap.read()
oH, oW, _ = img.shape

lett = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
          "U", "V", "W", "X", "Y", "Z"]
i = 0

folder = "data/"
counter = 0

def crop_hand(frame, landmarks):
    h, w, _ = frame.shape
    min_x, min_y, max_x, max_y = w, h, 0, 0

    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    fh, fw = max_y - min_y, max_x - min_x
    if (min_y - int(fh * offset / 2) >= 0 and
            max_y + int(fh * offset / 2) <= h and
            min_x - int(fw * offset / 2) >= 0 and
            max_x + int(fw * offset / 2) <= w):
        return hand_mesh(frame[min_y - int(fh * offset / 2):max_y + int(fh * offset / 2), min_x - int(fw * offset / 2): max_x + int(fw * offset / 2)])
    else:
        return None

def hand_mesh(frame):
    h, w, _ = frame.shape
    frameWhite = np.ones((int(2 * h / offset) - 10,int(2 * w /offset) - 10,3),np.uint8) * 255
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(frameWhite, handLms, mpHands.HAND_CONNECTIONS)
    return square_img(frameWhite)

def square_img(frame):
    imgWhite = np.ones((imgSize,imgSize,3),np.uint8) * 255

    h, w, _ = frame.shape

    try:
        aspectRatio = h /w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(frame,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300 - wCal) / 2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(frame,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        return imgWhite
    except e as Exception:
        print(e)
        return None
        

while True:
    success, img = cap.read()
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            imgWhite = crop_hand(img, handLms.landmark)
            if imgWhite is not None:
                cv2.imshow("ImageWhite", imgWhite)
            
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s") and imgWhite is not None and counter <= 550:
        counter += 1
        cv2.imwrite('forThePPT/cut31.jpg',img)
        cv2.imwrite('forThePPT/cut3.jpg',imgWhite)
        print(counter)
    if key == ord("n"):
        counter = 0
        i+=1
        print(lett[i])

cap.release()
cv2.destroyAllWindows()
    
