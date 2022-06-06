#Install 
pip install tensorflow_addons

#Import important packages
import numpy as np

import tensorflow as tf 
import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras import preprocessing

#Check Version by printing
print(tf.__version__)
print(tfa.__version__)

#Path dir
TRAIN_DIR = ('/gambar/Training/Training/')
TEST_DIR = ('/gambar/Testing/Testing/')


# read raw image file & 
#///// Train /////
# sedikit augmentation hanya untuk data train
batch_size = 64
train_datagen = preprocessing.image.ImageDataGenerator(
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                   target_size=(48,48),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

#///// validation /////
validation_datagen = preprocessing.image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(TEST_DIR,
                                                             target_size=(48,48),
                                                             batch_size=batch_size,
                                                             class_mode='categorical')

########MODELING PART##########
from keras.applications.efficientnet_v2 import EfficientNetV2B1, preprocess_input