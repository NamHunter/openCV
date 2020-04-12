import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
# gray = cv2.imread('image\Test\lena.jpg',0)
# frame = cv2.imread('image\Test\lena.jpg',1)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h,x:x+h]
        roi_color = frame[y:y+h,x:x+h]

        id,conf = recognizer.predict(roi_gray)
        if(conf >=45):
            print(id)
            print(labels[id])
        img_item =  'my-image.png'
        cv2.imwrite(img_item, roi_gray)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
