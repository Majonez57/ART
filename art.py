import numpy as np
import cv2

# The camera frame size and points
FRAMESIZE = 200
CAMERAFRAME = np.array([
    [0,0],
    [FRAMESIZE-1, 0],
    [FRAMESIZE-1, FRAMESIZE-1],
    [0, FRAMESIZE-1]], dtype="float32")

# Returns list of visible tag IDs in an Image
def simpleLocateTags(frame, drawOutline=False, drawZoom=False):
    tagCorners = findContours(frame) # The outer corners of the tags
    out = frame.copy()
    
    for corners in tagCorners:
        # Get the corners, and find the homography matrix
        c_rez = corners[:, 0]
        H_matrix = homograph(CAMERAFRAME, orderCorners(c_rez))
        # Apply homography matrix and colour change
        tag = cv2.warpPerspective(frame, H_matrix, (200, 200))
        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        # Get tag ID and Orientation. If these are none, we can ignore the detection
        decode, orientation = AR4Decode(tag1)
        
        if orientation != None and decode != None:
            #We want to make sure we are always looking the right way up!
            rotationMat = cv2.getRotationMatrix2D((100, 100), -orientation, 1.0)
            tag2 = cv2.warpAffine(tag1, rotationMat, tag1.shape[1::-1], flags=cv2.INTER_LINEAR)
        
            if drawZoom:
                t, zim = cv2.threshold(tag2, 150, 255, cv2.THRESH_BINARY)
                cv2.imshow(f"Zoomed Tag", zim)
            if drawOutline:
                cv2.putText(out, f"{decode}", c_rez[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.drawContours(out, [corners], -1, (0,255,0), 2)
    
    
    cv2.imshow("Outline", out)

# Decodes the tags ID. Returns value of tag and the Orientation
def AR4Decode(image):
    ret, image_bw = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY) #Apply threshold for binary image
    white = 255 #Bottom left corner should be white
    croppedImage = image_bw[50:150, 50:150] #Crop exterior black parts
    
    #Find middle pixel of each one of the ID pixels. We can then check the colour to extract the ID
    tl = '0' if croppedImage[37,37] == 255 else '1'
    bl = '0' if croppedImage[62,37] == 255 else '1'
    tr = '0' if croppedImage[37,62] == 255 else '1'
    br = '0' if croppedImage[62,62] == 255 else '1'
    
    # Now that we have the values, we need to find the orientation to get the correct ID value
    if croppedImage[85, 85] == white:
        return int("".join([bl, br, tr, tl]), 2), 0
    elif croppedImage[15, 85] == white:
        return int("".join([br, tr, tl, bl]), 2), 90
    elif croppedImage[15, 15] == white:
        return int("".join([tr, tl, bl, br]), 2), 180
    elif croppedImage[85, 85] == white:
        return int("".join([tl ,bl, br, tr]), 2), -90
    else:
        return None, None

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
    rectangle = np.zeros((4,2), dtype="float32")
    
    pSum = points.sum(axis=1)
    
    rectangle[0] = points[np.argmin(pSum)]
    rectangle[2] = points[np.argmax(pSum)]
    
    diff = np.diff(points, axis=1)
    
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]
    
    return rectangle

# Takes an image, and returns two arrays; The exterior corners, and interior corners of the Tag.
def findContours(frame):
    #Convert Input Image to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Apply Gaussian Blur & Canny Edge Detection
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    edge = cv2.Canny(blur, 75, 200)
    
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
    finaloTagCorners = []
    for corners in otags:
        if len(corners) == 4:
            area = cv2.contourArea(corners)
            if area > 300:
                finaloTagCorners.append(corners)
    
    return finaloTagCorners