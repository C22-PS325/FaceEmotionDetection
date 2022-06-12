##Import important packages
import preprocessingdata as pdt
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_addon as tfa # You need to install first if  you didnt have it
data_path = "Dataset"
data, labels = pdt.load_data(data_path)

# Split the dataset into two subsets (70%-30%). The first one will be used for training.
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, shuffle=True, random_state=3)
print("Train data shape")
print(X_train.shape)
print(y_train.shape)
# The second subset will be split into validation and test set (50%-50%).
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, train_size=0.5, shuffle=True, random_state=3)
print("Valid data shape")
print(X_valid.shape)
print(y_valid.shape)
print("Test data shape")
print(X_test.shape)
print(y_test.shape)

# Mapping the emotion-categories
emot_label = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise'}

# Initialize the training data augmentation object
train_generator = ImageDataGenerator(rotation_range=15,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              brightness_range=(.6, 1.2),
                              shear_range=.15,
                              height_shift_range=0.2,
                              horizontal_flip=True)



#Building model with EffienNetB0 This function returns a Keras image classification model, 
#optionally loaded with weights pre-trained on ImageNet.


def run_model():

    inputs = Input(shape=(224, 224, 3)) 
    base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                drop_connect_rate=0.33, input_tensor=inputs)
    #Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(.4)(x)
    # apply penalties on layer parameters or layer activity during optimization. 
    # These penalties are summed into the loss function that the network optimizes.
    
    outputs = Dense(6, activation='softmax',activity_regularizer=regularizers.L2(0.01))(x)
    model = Model(inputs, outputs)
    #Weight decay is a regularization technique by adding a small penalty
    model.compile(optimizer = tfa.optimizers.AdamW(learning_rate=1e-4,weight_decay=1e-5), 
              loss = "categorical_crossentropy", 
              metrics = ["accuracy"])

    return model

model.summary()



epcohs = 25
batch_size = 64
filepath = "/Untitled Folder/model.h5"


"""
ModelCheckpoint callback is used in conjunction with training using model.fit () to save a model or weights 
(in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved. 

"""


# Define the necessary callbacks
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks = [checkpoint

history = model.fit(trainAug.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=len(X_train) // batch_size,
                 validation_data=(X_valid, y_valid),epochs=epcohs, callbacks=callbacks)
             
             
