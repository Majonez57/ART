import cv2
import numpy as np

SIZE = 200
BORDERN = 2
PIXELN = 8
PIXELL = int(SIZE/PIXELN)

#Generate Base
base = np.zeros((SIZE, SIZE, 3))
bl = BORDERN*PIXELL
base[bl:SIZE-bl, bl:SIZE-bl] = (255, 255, 255)
#Add Orientation pixels
base[bl:bl+PIXELL, bl:bl+PIXELL] = (0, 0, 0)
base[SIZE-(bl+PIXELL):SIZE-bl, bl:bl+PIXELL] = (0, 0, 0)
base[bl:bl+PIXELL, SIZE-(bl+PIXELL):SIZE-bl] = (0, 0, 0)
#Generate all possible fellas
for i in range(0, 16):
    new = np.copy(base)
    num = i
    #1,2,4,8
    if num >= 8: #Left Bottom
        num -= 8
        new[SIZE-bl-2*PIXELL:SIZE-bl-PIXELL, bl+PIXELL:bl+2*PIXELL] = (0, 0, 0)
    if num >= 4:   #Right Bottom
        num -= 4
        new[SIZE-bl-2*PIXELL:SIZE-bl-PIXELL, SIZE-bl-2*PIXELL:SIZE-bl-PIXELL] = (0, 0, 0) 
    if num >= 2: #Right Top
        num -= 2
        new[bl+PIXELL:bl+2*PIXELL, SIZE-bl-2*PIXELL:SIZE-bl-PIXELL] = (0, 0, 0)
    if num >= 1: #Right Top
        new[bl+PIXELL:bl+2*PIXELL, bl+PIXELL:bl+2*PIXELL] = (0, 0, 0) 
    
    cv2.imshow("1", new)
    cv2.waitKey(0)

