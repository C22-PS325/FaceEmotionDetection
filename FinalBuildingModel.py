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
from keras.applications.mobilenet_v2 import MobileNetV2

# modelling
size = (128,128,3)

base_model=MobileNetV2(input_shape = size,weights="imagenet",include_top=False)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = layers.Dense(7, activation='softmax')(x)

# this is the model we will train
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)