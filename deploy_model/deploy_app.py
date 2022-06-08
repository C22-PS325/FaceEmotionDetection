from flask import Flask, render_template, request
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1