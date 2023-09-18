#images are in the folder: Chips_Thermal_Face_Dataset/images
#labels are in the folder: Chips_Thermal_Face_Dataset/annotations_yolo_format
#train a model to recognize if there is a person in the image or not
#use the model to predict if there is a person in the image or not

import os
import cv2
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential

# define the color range for detecting people
lower_color = (0, 0, 0)  # lower bound of the color range
upper_color = (255, 255, 255)  # upper bound of the color range

#load the images
images = []
for filename in os.listdir('Chips_Thermal_Face_Dataset/images'):
    img = cv2.imread(os.path.join('Chips_Thermal_Face_Dataset/images',filename))
    if img is not None:
        images.append(img)

#load the labels
labels = []
for filename in os.listdir('Chips_Thermal_Face_Dataset/annotations_yolo_format'):
    label = np.load(os.path.join('Chips_Thermal_Face_Dataset/annotations_yolo_format',filename))
    if label is not None:
        labels.append(label)

#resize the images and labels
for i in range(len(images)):
    if i >= len(labels):
        raise ValueError('Missing label for image')
    if images[i].shape[:2] != labels[i].shape[:2]:
        raise ValueError('Image and label dimensions do not match')
    images[i] = cv2.resize(images[i], (224, 224))
    labels[i] = cv2.resize(labels[i], (224, 224))

#convert the labels to binary
for i in range(len(labels)):
    if lower_color is not None and upper_color is not None:
        mask = cv2.inRange(labels[i], lower_color, upper_color)
        labels[i] = mask / 255
    else:
        labels[i] = (labels[i] > 0).astype('float32')

#normalize the images
images = np.array(images)
images = images.astype('float32')
images /= 255

#split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

#build the model
model = Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
model.fit(train_images, train_labels, epochs=2, batch_size=64)

#evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#save the model
model.save('thermal_model.h5')