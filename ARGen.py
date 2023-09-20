import cv2
import numpy as np

PIXELL = 40
BORDERN = 2
DATABITS = 4
PIXELN = 2*BORDERN + DATABITS + 2 + 2 #Add two for the orientation ring, 2 for the border
OSIZE = PIXELL*PIXELN
SIZE = PIXELL*(PIXELN-2)

def getTag(ID): #Returns np array containing the AR tag for the given ID
    totalSize = OSIZE
    #Generate Base
    base = np.zeros((totalSize , totalSize , 3))
    base[:] = (255,255,255) # Outer Border

    base[PIXELL:SIZE+PIXELL, PIXELL:SIZE+PIXELL] = (0,0,0) #Outer Dark Part

    bl = (BORDERN+1)*PIXELL #Width of the Border
    base[bl:OSIZE-bl, bl:OSIZE-bl] = (255, 255, 255) # Inner Part
    
    #Add Orientation pixels
    #bl += PIXELL
    base[bl:bl+PIXELL, bl:bl+PIXELL] = (0, 0, 0)
    base[totalSize-(bl+PIXELL):totalSize-bl, bl:bl+PIXELL] = (0, 0, 0)
    base[bl:bl+PIXELL, totalSize-(bl+PIXELL):totalSize-bl] = (0, 0, 0)

    ##Now we have the blank tag. We must now add data.
    code = f"{ID:b}"
    code = (DATABITS**2 - len(code))*'0' + code
    print(code)
    # We now have a 16 bit number for our ID
    for i in range(len(code)):
        if code[i] == '1': # 1s will be black pixels
            pp = pPos(i)
            print(pp)
            y = (pp[0]+1)*PIXELL + bl
            x = (pp[1]+1)*PIXELL + bl

            base[x:x+PIXELL, y:y+PIXELL] = (0, 0, 0)
    
    return base

# Helper function that returns a tuple with the grid coordinates of a position in a snake
def pPos(position, gridS=DATABITS):
    res = [0,0]

    res[0] = position % DATABITS
    res[1] = position // DATABITS

    return(res)

n = ((0b1001 << 12) + 
    (0b0000 << 8) +
    (0b1001 << 4) +
    (0b0110))


show = getTag(1+1024)
cv2.imshow(".", show)
cv2.waitKey(0)
