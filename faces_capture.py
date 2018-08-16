import numpy as np
import cv2
import pickle
import os

#cascade classifier
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
img_no = 0

name = input('Enter the name: ')

while(img_no < 100):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #getting a grayscale frame
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        
        #x,y,w,h - region of interest x,y - cordinates , w,h - width and height
        color = (0, 0, 255) #BGR
        stroke = 2
        #cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        cv2.rectangle(frame, (x-20, y-100), (x+w+20, y+h+50), color, stroke)
        roi_color = frame[y-100:y+h+50, x-20:x+w+20] #region of interest in color image
        #name = "Namal Priyadarshani"
        path = "D:/COM/Other/Machine Learning/OpenCV/Face_Identification/Images/"+name
        img = "D:/COM/Other/Machine Learning/OpenCV/Face_Identification/Images/"+name+"/"+str(img_no)  + ".png"

        if not os.path.exists(path):
            os.makedirs(path)

        cv2.imwrite(img, roi_color)
        img_no += 1

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()