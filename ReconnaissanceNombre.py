# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:15:35 2020

@author: Plante Matthieu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.utils import np_utils
import keras.backend.tensorflow_backend as tfback


def _get_available_gpus():
    #Get a list of available gpu devices (formatted as strings).

    # Returns
        #0A list of available GPU devices.
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



"""
CREATE MODEL
"""

"""

(images, targets), (images_test, targets_test) = tf.keras.datasets.mnist.load_data()



"""
#Normalize data
"""

images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1).astype('float32')

images = images/255
images_test = images_test/255


targets = np_utils.to_categorical(targets)
targets_test = np_utils.to_categorical(targets_test)
num_classes = targets.shape[1]


"""
#Model
"""
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu', data_format='channels_last'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model




"""
#Compile the Model
"""

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(images, targets, epochs=10, validation_split = 0.2)


"""
#Check the courbs of loss and accuracy
"""

loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

loss_val_curve = history.history["val_loss"]
acc_val_curve = history.history["val_accuracy"]

plt.plot(loss_curve, label="Train")
plt.plot(loss_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.show()

plt.plot(acc_curve, label="Train")
plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()

model.evaluate(images_test, targets_test)

model.save('reco_nombre_simple_model.h5')


"""

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('reco_nombre_simple_model.h5')

# Show the model architecture
new_model.summary()


"""
#Webcam
"""
cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    cv2.imshow("Number detection", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "nb.png"
        #Enregistre la photo
        cv2.imwrite(img_name, frame)
        break

cam.release()

cv2.destroyAllWindows()

#Read the file
img = np.array(cv2.imread(img_name))

rgb_weights = [0.2989, 0.5870, 0.1140]

#Transform to greyscale and normaize
img_grey = np.dot(img[...,:3], rgb_weights)
img_grey = cv2.resize(img_grey, (28, 28))

#Preprocess on the image 
for i in range (4):
    for j in range (28):
        img_grey[i][j] = 0
        
for i in range (24, 28):
    for j in range (28):
        img_grey[i][j] = 0

for i in range (4, 24):
    for j in range (28):
        if (img_grey[i][j] > 100):
            img_grey[i][j] = 0
        else :
            img_grey[i][j] += 50


#Display
plt.imshow(img_grey, cmap = "gray")


img_flatten = np.array([img_grey]).reshape(28, 28, 1)/255

#Predictions
pred = new_model.predict(np.array([img_flatten]))

maxim = pred[0][0]
maxim1 = 0

for i in range (1, 10):
    if pred[0][i] > maxim:
        maxim = pred[0][i]
        maxim1 = i
        
print("The predicted number is: {} w/ a probability of: {}%".format(maxim1, round(maxim*100, 2)))
