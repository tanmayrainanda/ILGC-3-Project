import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.regularizers import l2

# dimensions of our images.
img_width, img_height = 32,24

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
num_classes = 4  # Change this to the number of classes
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Load the weights
model.load_weights('scale_down_weights_v3.h5')

# Load the image file
#img_path = '/Users/tanmay/Downloads/Screenshot 2023-11-29 at 1.05.34 AM.png'
img_path = 'thermal_image_41.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))

# Convert the image to a numpy array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

class_labels = {'Cold': 0, 'Hot': 1, 'Neutral': 2, 'Very Hot': 3}

# Predict the class of the image
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    proba = model.predict(images, batch_size=10)
    for i, class_label in enumerate(class_labels.keys()):
        print(f"Probability of {class_label}: {proba[0][i]}")
    classes = np.argmax(proba, axis=-1)
    class_label = list(class_labels.keys())[list(class_labels.values()).index(classes[0])]
    return class_label

print(classify_image(img_path))