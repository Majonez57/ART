import cv2
import art
from time import sleep as zzz


camera_port = 0
camera = cv2.VideoCapture(camera_port)

while True:
    
    result, image = camera.read()

    if result:
        
        art._testFunc(image)
        tags = art.findTags(image)
        for tagvalue, corners in tags:
            
            cv2.polylines(image, [corners], True, (50,0,200), 2)
            cv2.putText(image, f"{tagvalue}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("RESULT", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("No image detected.")

camera.release()
cv2.destroyAllWindows()