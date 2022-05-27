# Menginstall packages yang penting
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Untuk menampilkan hasil versi Tenserflow 
print("Tensorflow version:", tf.__version__)

#Untuk ekstraksi zip file
import zipfile
with zipfile.ZipFile("D:\BANGKIT\FaceEmotionDetection\gambar\gambar.zip", 'r') as zip_ref:
     zip_ref.extractall("D:\BANGKIT\FaceEmotionDetection\gambar\gambar.zip")
        
#Untuk menmapilkan jumlah gambar dari tiap kategori
for emotion in os.listdir("D:\BANGKIT\FaceEmotionDetection\gambar\Training\"):
    print(str(len(os.listdir("D:\BANGKIT\FaceEmotionDetection\gambar\Training\ " + emotion))) + " " + emotion + " images")
    

#Menentukan gambar Weight, Height , color     
img_size =  (48,48,3)
batch_size = 64

#melakukan normalisasi dengan menggunakan fungsi ImageDataGenerator
data_train = ImageDataGenerator(horizontal_flip=True)

                          
train_gen = data_train.flow_from_directory("D:\BANGKIT\FaceEmotionDetection\gambar\Training\",
                                                    target_size=(img_size,img_size),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

data_val = ImageDataGenerator(horizontal_flip=True)
val_gen = data_val.flow_from_directory("D:\BANGKIT\FaceEmotionDetection\gambar\Testing\",
                                                    target_size=(img_size,img_size),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)