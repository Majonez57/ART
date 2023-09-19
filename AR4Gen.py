import cv2
import numpy as np

SIZE = 200
BORDERN = 2
PIXELN = 8
PIXELL = int(SIZE/PIXELN)

#Generate Base
base = np.zeros((SIZE, SIZE, 3))
borderLength = BORDERN*PIXELL
base[borderLength:SIZE-borderLength, borderLength:SIZE-borderLength] = (255, 255, 255)
#Add Orientation pixels
base[borderLength:borderLength+PIXELL, borderLength:borderLength+PIXELL] = (0, 0, 0)
base[SIZE-(borderLength+PIXELL):SIZE-borderLength, borderLength:borderLength+PIXELL] = (0, 0, 0)
base[borderLength:borderLength+PIXELL, SIZE-(borderLength+PIXELL):SIZE-borderLength] = (0, 0, 0)
#Generate all possible fellas



cv2.imshow(".", base)
cv2.waitKey(0)

