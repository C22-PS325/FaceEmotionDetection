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

#metode ini mengambil jalur direktori dan menghasilkan batch data yang ditambah                        
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
                          
                        
########### CNN Model ############# 
#Model Building VGG16 adalah arsitektur convolution neural net (CNN ) yang digunakan untuk memenangkan kompetisi ILSVR(Imagenet) pada tahun 2014. Ini dianggap sebagai salah satu arsitektur model visi yang sangat baik hingga saat ini. Hal yang paling unik tentang VGG16 adalah bahwa alih-alih memiliki sejumlah besar hyper-parameter, mereka berfokus pada memiliki lapisan konvolusi filter 3x3 dengan langkah 1 dan selalu menggunakan lapisan padding dan maxpool yang sama dari filter 2x2 langkah 2. Ini mengikuti pengaturan ini convolution dan max pool layer secara konsisten di seluruh arsitektur. Pada akhirnya memiliki 2 FC (lapisan yang terhubung penuh) diikuti oleh softmax untuk output. 16 di VGG16 mengacu pada itu memiliki 16 lapisan yang memiliki bobot. Jaringan ini adalah jaringan yang cukup besar dan memiliki sekitar 138 juta (perkiraan) parameter.
                          
from keras.applications.vgg16 import VGG16

base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=img_size
    )

base_model.summary()