# Menginstall packages yang penting
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Untuk menampilkan hasil versi Tenserflow 
print("Tensorflow version:", tf.__version__)


train_path = ('/content/drive/MyDrive/Colab Notebooks/Bangkit/gambar/Training/Training/')
test_path = ('/content/drive/MyDrive/Colab Notebooks/Bangkit/gambar/Testing/Testing/')

#Untuk menmapilkan jumlah gambar dari tiap kategori
for emotion in os.listdir("D:\BANGKIT\FaceEmotionDetection\gambar\Training\"):
    print(str(len(os.listdir("D:\BANGKIT\FaceEmotionDetection\gambar\Training\ " + emotion))) + " " + emotion + " images")
    

#Menentukan gambar Weight, Height , color     
img_size =  (48,48,1)
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

my_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=img_size
    )

my_model.summary()

CLASSES = 6

model = Sequential()
model.add(my_model)
model.add(Flatten())
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(CLASSES, activation="softmax"))
                          
from tensorflow.keras.optimizers import RMSprop
def deep_model(model, X_train, Y_train, epochs, batch_size):
   
    model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy'])
    
    history = model.fit(X_train
                       , Y_train
                       , epochs=epochs
                       , batch_size=batch_size
                       , verbose=1)
    return history

from tqdm import tqdm
import cv2
import numpy
def load_data(dir_path, img_size):
   
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    gmbr = cv2.imread(dir_path + path + '/' + file)
                    gmbr = gmbr.astype('float32') / 255
                    resized = cv2.resize(gmbr, img_size,  interpolation = cv2.INTER_AREA)
                    X.append(resized)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels
                          
                          
                          
X_train, y_train, train_labels = load_data(train_path, img_size)
X_test, y_test, test_labels = load_data(test_path,img_size)
                          
epochs = 40
batch_size = 64

history = deep_model(model, X_train, Y_train, epochs, batch_size)
                          
#Menampilkan Hasil prediksi emotion                           
from random import randint

l = len(namafiles)
base_path = test_path
for i in range(10):  # 10 images
    
    rnd_number = randint(0,l-1)
    namafile,pred_class,actual_class = pred_result.loc[rnd_number]
    
    img_path = os.path.join(base_path,namafile)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Predicted Class: {} {} Actual Class: {}".format(pred_class,'\n',actual_class))
    plt.show()
    pass

