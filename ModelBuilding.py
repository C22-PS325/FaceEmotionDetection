# Menginstall packages yang penting
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Untuk menampilkan hasil versi Tenserflow 
print("Tensorflow version:", tf.__version__)


TRAIN_DIR = ('/content/drive/MyDrive/Colab Notebooks/Bangkit/gambar/Training/Training/')
TEST_DIR = ('/content/drive/MyDrive/Colab Notebooks/Bangkit/gambar/Testing/Testing/')

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
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

data_val = ImageDataGenerator(horizontal_flip=True)
val_gen = data_val.flow_from_directory("D:\BANGKIT\FaceEmotionDetection\gambar\Testing\",
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
                          
 
########### CNN Model ############# 
#Model Building VGG16 adalah arsitektur convolution neural net (CNN ) yang digunakan untuk memenangkan kompetisi ILSVR(Imagenet) pada tahun 2014. Ini dianggap sebagai salah satu arsitektur model visi yang sangat baik hingga saat ini. Hal yang paling unik tentang VGG16 adalah bahwa alih-alih memiliki sejumlah besar hyper-parameter, mereka berfokus pada memiliki lapisan konvolusi filter 3x3 dengan langkah 1 dan selalu menggunakan lapisan padding dan maxpool yang sama dari filter 2x2 langkah 2. Ini mengikuti pengaturan ini convolution dan max pool layer secara konsisten di seluruh arsitektur. Pada akhirnya memiliki 2 FC (lapisan yang terhubung penuh) diikuti oleh softmax untuk output. 16 di VGG16 mengacu pada itu memiliki 16 lapisan yang memiliki bobot. Jaringan ini adalah jaringan yang cukup besar dan memiliki sekitar 138 juta (perkiraan) parameter.

from keras.applications.vgg16 import VGG16

model = VGG16(
        weights=None,
        include_top=False,
        input_shape=img_size
    )

model.summary()
                          
                          
X_train, y_train, train_labels = load_data(TRAIN_DIR, img_size)
X_test, y_test, test_labels = load_data(TEST_DIR,img_size)
                          
epochs = 40
batch_size = 64

history = deep_model(model, X_train, Y_train, epochs, batch_size)
                          
#Menampilkan Hasil prediksi emotion                           
from random import randint

l = len(filenames)
base_path = TEST_DIR
for i in range(10):  # 10 images
    
    rnd_number = randint(0,l-1)
    filename,pred_class,actual_class = pred_result.loc[rnd_number]
    
    img_path = os.path.join(base_path,filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Predicted Class: {} {} Actual Class: {}".format(pred_class,'\n',actual_class))
    plt.show()
    pass

