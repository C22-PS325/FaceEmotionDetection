from flask import Flask, render_template, request
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

#Read load file picture
image = cv2.imread('emot.jpg')

# Object Detection Algorithm used to identify face in an image or a real time video

#Loading the required haar-cascade XML classifier file
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying the face detection method on the grayscale image
wajah = cascade.detectMultiScale(gray,1.1, 3)