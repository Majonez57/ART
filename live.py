import cv2
import art
from time import sleep as zzz


camera_port = 0
camera = cv2.VideoCapture(camera_port)

while True:
    
    result, image = camera.read()

    if result:
        
        art.simpleLocateTags(image, True, True)
        
        # cv2.drawContours(image, tagCor, -1, (50,0,200), 2)
        
        # cv2.imshow("cam", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("No image detected.")

camera.release()
cv2.destroyAllWindows()