import cv2
import urllib.request
import numpy as np

while True:
    req = urllib.request.urlopen('http://192.168.43.47/capture')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'

    cv2.imshow('random_title', img)
    
    # Refresh the image after 100 milliseconds
    if cv2.waitKey(100) & 0xff == 27:
        break  # Exit the loop if the 'Esc' key is pressed

cv2.destroyAllWindows()
