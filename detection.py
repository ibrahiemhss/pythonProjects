import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#use trained xml
face_cascade=cv2.CascadeClassifier('face.xml')
eye_cascade=cv2.CascadeClassifier('eyes.xml')
mouth_cascade=cv2.CascadeClassifier("mouth.xml")

while 1:

    #capture frame by frame
    ret, img = cap.read()
    #convert the video into gray video without color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect face,eyes and moouth in the video
    faces = face_cascade.detectMultiScale(gray, 50, 50)
    #Draw a rectangle boxes around face ,eye and mouth
    for(x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_color)
        for(mx,my,mw,mh) in mouth:
            cv2.rectangle(roi_color, (mx,my), (mx+mw,my+mh),(0,0,255), 5)

            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

                #Display the resuling frame
                cv2.imshow('Frame',img)
                k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()    

