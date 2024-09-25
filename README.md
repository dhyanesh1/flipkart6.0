    # flipkartgrid6.0
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d dhyaneshwaranm/fruit-freshness

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle


import zipfile
zip_ref = zipfile.ZipFile('/content/fruit-freshness.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

list_dir = []
def load_rand():
    X=[]
    dir_path='dataset/Train'
    for sub_dir in tqdm(os.listdir(dir_path)):
        print(sub_dir)
        list_dir.append(sub_dir)
        path_main=os.path.join(dir_path,sub_dir)
        i=0
        for img_name in os.listdir(path_main):
            if i>=6:
                break
            img=cv2.imread(os.path.join(path_main,img_name))
            img=cv2.resize(img,(100,100))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            X.append(img)
            i+=1
    return X


dir_path='dataset/Train'
dir_path_test='dataset/Test'

X=load_rand()

len(list_dir)

X=np.array(X)
X.shape

def show_subpot(X,title=False,Y=None):
    if X.shape[0]==36:
        f, ax= plt.subplots(6,6, figsize=(40,60))
        list_fruits=list_dir
        for i,img in enumerate(X):
            ax[i//6][i%6].imshow(img, aspect='auto')
            if title==False:
                ax[i//6][i%6].set_title(list_fruits[i//6])
            elif title and Y is not None:
                ax[i//6][i%6].set_title(Y[i])
        plt.show()
    else:
        print('Cannot plot')

del X

def load_rottenvsfresh():
    quality=['fresh', 'rotten']
    X,Y=[],[]
    z=[]
    for cata in tqdm(os.listdir(dir_path)):
        if quality[0] in cata:
            path_main=os.path.join(dir_path,cata)
            for img_name in os.listdir(path_main):
                img=cv2.imread(os.path.join(path_main,img_name))
                img=cv2.resize(img,(100,100))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                z.append([img,0])
        else:
            path_main=os.path.join(dir_path,cata)
            for img_name in os.listdir(path_main):
                img=cv2.imread(os.path.join(path_main,img_name))
                img=cv2.resize(img,(100,100))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                z.append([img,1])
    print('Shuffling your data.....')
    shuffle(z)
    for images, labels in tqdm(z):
        X.append(images);Y.append(labels)
    return X,Y

X,Y=load_rottenvsfresh()

Y=np.array(Y)
X=np.array(X)

import tqdm # import the tqdm module
dir_path='dataset/Train'
dir_path_test='dataset/Test'
def load_rottenvsfresh_valset():
    quality=['fresh', 'rotten']
    X,Y=[],[]
    z=[]
    for cata in tqdm.tqdm(os.listdir(dir_path)):
        if quality[0] in cata:
            path_main=os.path.join(dir_path_test,cata)
            for img_name in os.listdir(path_main):
                img=cv2.imread(os.path.join(path_main,img_name))
                img=cv2.resize(img,(100,100))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                z.append([img,0])
        else:
            path_main=os.path.join(dir_path_test,cata)
            for img_name in os.listdir(path_main):
                img=cv2.imread(os.path.join(path_main,img_name))
                img=cv2.resize(img,(100,100))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                z.append([img,1])
    print('Shuffling your data.....')
    shuffle(z)
    for images, labels in tqdm.tqdm(z):
        X.append(images);Y.append(labels)
    return X,Y


X_val,Y_val=load_rottenvsfresh_valset()

import pandas as pd
import matplotlib.pyplot as plt
Y_val=np.array(Y_val)
X_val=np.array(X_val)
y_ser=pd.Series(Y_val)
y_ser.value_counts()
plt.imshow(X_val[0])
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense



# Load the pre-trained MobileNetV2 model
mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the layers in the MobileNetV2 model
for layer in mobilenetv2_model.layers:
    layer.trainable = False

# Create a new Sequential model
model = Sequential()

# Add the MobileNetV2 model to the new model (up to the last convolutional layer)
model.add(mobilenetv2_model)

# Add the rest of the custom layers
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), depthwise_initializer='he_uniform', padding='same', activation='relu'))
model.add(SeparableConv2D(64, (3, 3), depthwise_initializer='he_uniform', padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, 100, 100, 3))

# Print the summary of the model
model.summary()




import keras

import tensorflow as tf
lr_rate=keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=6, verbose=1, mode='max',
    min_lr=0.00002, cooldown=2)

check_point=tf.keras.callbacks.ModelCheckpoint(
    filepath='modelcheckpt.keras', monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='min')


history=model.fit(X,Y,batch_size=16,validation_data=(X_val,Y_val),epochs= 30,
                 callbacks=[check_point])

import matplotlib.pyplot as plt
plt.figure(1, figsize = (20, 12))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot( history.history["loss"], label = "Training Loss")
plt.plot( history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot( history.history["accuracy"], label = "Training Accuracy")
plt.plot( history.history["val_accuracy"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()

import numpy as np
model.evaluate(X_val,Y_val)

model.save('rottenvsfresh.h5')

from keras.models import Model, load_model

pip install --upgrade keras

pip install --upgrade tensorflow

import numpy as np
import os

# Check if the files exist
if os.path.exists('X_val.npy') and os.path.exists('Y_val.npy'):
    # Load the validation data from .npy files
    X_val = np.load('X_val.npy')  # Changed extension to .npy
    Y_val = np.load('Y_val.npy')  # Changed extension to .npy

    new_model.evaluate(X_val,Y_val)
else:
    print("Error: 'X_val.npy' or 'Y_val.npy' not found in the current directory.")

import numpy as np
import os
from keras.models import load_model # Import load_model



import matplotlib.pyplot as plt # imports the matplotlib.pyplot module and sets it to the alias plt
# print(X_val[0])
if X_val is not None:
    plt.imshow(X_val[155])
    plt.show()
else:
    print("X_val is None. Please make sure 'X_val.npy' exists and is loaded correctly.")

model.predict(X_val[155].reshape(1,100,100,3))

import cv2
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.layers import Activation
import tensorflow as tf

# Classify fresh/rotten
def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    if res < threshold_fresh:
        print("The item is FRESH!")
    elif threshold_fresh < res < threshold_medium:
        print("The item is MEDIUM FRESH")
    else:
        print("The item is NOT FRESH")


def pre_proc_img(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Preprocess the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Add an extra dimension for the batch
    return img


def evaluate_rotten_vs_fresh(image_path):
    # Use CustomObjectScope to handle potential custom objects or layers
    with CustomObjectScope({'Activation': Activation}):
      from keras.models import load_model
    from keras.utils import CustomObjectScope
    from keras.layers import Activation

    # Define the custom objects
    custom_objects = {'Activation': Activation}
    model = load_model('rottenvsfresh.h5')

img_path = '/content/aa.jpeg'
img = pre_proc_img(img_path) # Preprocess the image
# is_rotten = evaluate_rotten_vs_fresh(img_path)
is_rotten = model.predict(img)[0][0] # Predict using the preprocessed image
print(f'Prediction: {is_rotten}')
print_fresh(is_rotten)
plt.imshow(img[0])
plt.show()
