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
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

data_val = ImageDataGenerator(horizontal_flip=True)
val_gen = data_val.flow_from_directory("D:\BANGKIT\FaceEmotionDetection\gambar\Testing\",
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
                          

                          
#Model Building VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.          
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