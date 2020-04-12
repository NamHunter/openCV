import os
from PIL import Image
import numpy as np 
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imagedir = os.path.join(BASE_DIR,"image\image RE")

face_cascade = cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cur_ID = 0
label_ids = {}
x_trains = []
y_labels = []

for root, dirs, files in os.walk(imagedir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()

            if not label in label_ids:
                label_ids[label] = cur_ID
                cur_ID += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L") #grayscale
            Image_array = np.array(pil_image, np.uint8)
            faces = face_cascade.detectMultiScale(Image_array,scaleFactor=1.5, minNeighbors=5)
            print(path)
            print(faces)
            for (x,y,w,h) in faces:
                roi = Image_array[y:y+h,x:x+w]
                x_trains.append(roi)
                y_labels.append(id_)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

print(label_ids)
recognizer.train(x_trains, np.array(y_labels))
recognizer.save("trainer.yml")