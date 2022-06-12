import os
import cv2
import numpy as np 
from keras.models import load_model


image_model = load_model('/content/Untitled Folder/model.h5')
image_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/haarcascade_frontalface_alt2.xml')
size = 224
image = cv2.imread('/content/S010_002_00000014.png')


faces = image_cascade.detectMultiScale(image,1.1, 3)

for x,y,w,h in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cropped = image[y:y+h, x:x+w]


image = cv2.resize(image, (size,size), interpolation=cv2.INTER_AREA)
image = image/255

image = image.reshape(1,size,size,-1)

pred = image_model.predict(image)
print(pred)

label_emotion =  ['sadness', 'happy', 'fear', 'suprise','disgust','anger']
for i in range(len(pred)):
    pred[i] = pred[i]*10
print(pred)
pred = np.argmax(pred)
print(pred)
final_pred = label_emotion[pred]

print(final_pred)