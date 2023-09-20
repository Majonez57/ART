import numpy as np
import cv2

# The tag frame size 
FRAMESIZE = 480
PIXELSIZE = 40
# Camera Frame Size
CAMERAFRAME = np.array([
    [0,0],
    [FRAMESIZE-1, 0],
    [FRAMESIZE-1, FRAMESIZE-1],
    [0, FRAMESIZE-1]], dtype="float32")

# Returns a list of tuples, each with the detected tag ID, and its 4 corners on the camera image
def findTags(frame, tagSize=FRAMESIZE-PIXELSIZE*2):
    tagCorners = findContours(frame) # The outer corners of the tags
    
    found = []
    
    for corners in tagCorners:
        # Get the corners, and find the homography matrix
        c_rez = corners[:, 0]
        H_matrix = homograph(CAMERAFRAME, orderCorners(c_rez))
        # Apply homography matrix and colour change
        tag = cv2.warpPerspective(frame, H_matrix, (tagSize, tagSize))
        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        # Get tag ID and Orientation. If these are none, we can ignore the detection
        decode, orientation = AR4Decode(tag1)
        if orientation != None and decode != None:
            found.append((decode, orderCorners(c_rez)))
    
    return found

# Displays image with detected tags highlighted and their IDs displayed
def highlightTags(frame, tagSize=FRAMESIZE):
    tagCorners = findContours(frame) # The outer corners of the tags
    out = frame.copy()
    
    for corners in tagCorners:
        # Get the corners, and find the homography matrix
        c_rez = corners[:, 0]
        H_matrix = homograph(CAMERAFRAME, orderCorners(c_rez))
        # Apply homography matrix and colour change
        tag = cv2.warpPerspective(frame, H_matrix, (tagSize, tagSize))
        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        # Get tag ID and Orientation. If these are none, we can ignore the detection
        decode, orientation = AR4Decode(tag1)
        
        if orientation != None and decode != None:
            #We want to make sure we are always looking the right way up!
            rotationMat = cv2.getRotationMatrix2D((tagSize//2, tagSize//2), -orientation, 1.0)
            tag2 = cv2.warpAffine(tag1, rotationMat, tag1.shape[1::-1], flags=cv2.INTER_LINEAR)
        
            t, zim = cv2.threshold(tag2, 180, 255, cv2.THRESH_BINARY)
            cv2.imshow(f"Zoomed Tag ~{decode}", zim)
            
            cv2.putText(out, f"{decode}", c_rez[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.drawContours(out, [corners], -1, (0,255,0), 2)
    
    
    cv2.imshow("Outline", out)

# Decodes the tags ID. Returns value of tag and the Orientation
def AR4Decode(image):
    def calvalue(pixels): #Calculates the total value of the ID in the correct orientation
        total = 0
        for i in range(4):
            for j in range(4):
                if pixels[i][j] != 255: #Not White
                    print(i*4 + j)
                    total += 2**(15 - (i*4 + j))
        return total

    ret, image_bw = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY) #Apply threshold for binary image. Result is black/white
    white = 255 #Bottom left corner should be white
    border = int(PIXELSIZE*2)
    croppedImage = image_bw[border:FRAMESIZE-border, border:FRAMESIZE-border] #Crop exterior black parts
    
    # First we want to find the orientation pixels
    l = int(PIXELSIZE)
    h = int(PIXELSIZE*7)

    if croppedImage[h, h] == white: ##Normal way up!
        pass 
    elif croppedImage[l, h] == white:
        croppedImage = cv2.rotate(croppedImage, cv2.ROTATE_90_CLOCKWISE)
    elif croppedImage[l, l] == white:
        croppedImage = cv2.rotate(croppedImage, cv2.ROTATE_180)
    elif croppedImage[h, l] == white:
        croppedImage = cv2.rotate(croppedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None, None   
    
    #showImage = croppedImage.copy()

    #Find middle pixel of each one of the ID pixels. We can then check the colour to extract the ID
    # TODO use average value of ID pixel for more reliable ID
    pixelValss = []
    for i in range(4):
        row = []
        for j in range(4):
            x = int((PIXELSIZE)*(i+2.5))
            y = int((PIXELSIZE)*(j+2.5))
            row.append(croppedImage[x,y])
            #cv2.circle(showImage,(x,y), 2, (0,255,0), -1)
        
        pixelValss.append(row)
    
    #cv2.imshow(".", showImage)

    v = calvalue(pixelValss)
    return v, ord

# Computers the Homography Matrix between world and camera frame
def homograph(p, p1):
    A = []
    p2 = orderCorners(p)
    
    for i in range(0, len(p1)):
        # Ew Matrix maths
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h

# Returns the order of points in the camera frame
def orderCorners(points):
    rectangle = np.zeros((4,2), dtype="int32")
    
    pSum = points.sum(axis=1)
    
    rectangle[0] = points[np.argmin(pSum)]
    rectangle[2] = points[np.argmax(pSum)]
    
    diff = np.diff(points, axis=1)
    
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]
    
    return rectangle

# Takes an image, and returns the exterior corners
def findContours(frame):
    #Convert Input Image to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Apply Gaussian Blur & Canny Edge Detection
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    edge = cv2.Canny(blur, 100, 200)
    
    # Use cv2 to find all contours
    conts, h = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # We don't care about contours that are too large, or those that have other countours within them
    index = []
    for heirarchy in h[0]:
        if heirarchy[3] != -1: # If contour has no parents we remove it
            index.append(heirarchy[3])
    
    # Check each contour for viability (How likely it is to be an AR tag)
    otags = []
    for i in index:
        perimeter = cv2.arcLength(conts[i], True) #Perimeter of the Contour
        
        #Approximating the contour into a closed polygon
        accuracy = 0.02*perimeter #2% of perimeter as the maximum approximation
        innerCorners = cv2.approxPolyDP(conts[i], accuracy, True) 
        
        if len(innerCorners) > 4: #If approximation has more than 4 edges it is likely the inner tag shape!
            #This means that it's parent will be the edges of the whole tag! We repeat as above:
            OuterPerimeter = cv2.arcLength(conts[i-1], True) #Note the c-1 for the parent
            OuterAccuracy = 0.02*OuterPerimeter
            outerCorners = cv2.approxPolyDP(conts[i-1], OuterAccuracy, True) 
            otags.append(outerCorners)
    
    # Just to be sure, we will make sure all our tags have 4 corners, and are no bigger than a experimental constant
    tagCorners = []
    for corners in otags:
        if len(corners) == 4:
            area = cv2.contourArea(corners)
            if area > 300:
                tagCorners.append(corners)
    
    return tagCorners