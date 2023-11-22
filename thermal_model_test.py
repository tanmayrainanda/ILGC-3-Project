import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Load the dataset
image_dir = 'Collected Dataset'
label_dir = 'Test_Labels'
test_image_dir = 'Collected Dataset'
test_label_dir = 'Test_Labels'

# Preprocess the images
def preprocess_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize the image
            img = np.expand_dims(img, axis=-1)  # Add an extra dimension for the channel
            images.append(img)
    return np.array(images)

# Preprocess the labels
def preprocess_labels(label_dir):
    labels = []
    for filename in os.listdir(label_dir):
        with open(os.path.join(label_dir, filename), 'r') as f:
            # Assume the label is the first number in the file
            label = float(f.read().strip().split()[0])
            labels.append(label)
    return np.array(labels)

# Load and preprocess the dataset
images = preprocess_images(image_dir)
labels = preprocess_labels(label_dir)

# Define the model
model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(pool_size=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(11, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10)

# Evaluate the model
test_images = preprocess_images(test_image_dir)
test_labels = preprocess_labels(test_label_dir)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

def classify_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize the image
        img = np.expand_dims(img, axis=-1)  # Add an extra dimension for the channel
        img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch size
        prediction = model.predict(img)
        temp_status = np.argmax(prediction[0, :3]) - 1  # First 3 values for temperature status
        temp_adjust = np.argmax(prediction[0, 3:]) - 5  # Remaining values for temperature adjustment
        return temp_status, temp_adjust


image_path = "/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/FLIR0023.jpg"

print(classify_image(image_path))