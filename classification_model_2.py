import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

# dimensions of our images.
img_width, img_height = 32,24

data_dir = '/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/scaled_down'
label_file = '/Users/tanmay/Documents/GitHub/ILGC-3-Project/image_data_labelled_2.csv'
epochs = 200
batch_size = 16

# Load the labels from the CSV file
df = pd.read_csv(label_file)

# Split the data into training and validation sets
train_df, validation_df = train_test_split(df, test_size=0.2)

train_df['id'] = train_df['id'].astype(str)
validation_df['id'] = validation_df['id'].astype(str)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Define a more aggressive data augmentation configuration for "cold" class
cold_augmentation = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,  # Additional rotation
    width_shift_range=0.2,  # Additional horizontal shift
    height_shift_range=0.2,  # Additional vertical shift
    brightness_range=[0.5, 1.5]  # Additional brightness adjustment
)

# Apply the aggressive augmentation only to the "cold" class
train_generator_cold_augmented = cold_augmentation.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="img_name",
    y_col="choice",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Apply the regular augmentation to the other classes
train_generator_regular = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="img_name",
    y_col="choice",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=data_dir,
    x_col="img_name",
    y_col="choice",
    target_size=(img_width, img_height),  # Adjusted size
    batch_size=batch_size,
    class_mode='categorical')

# Create a single generator that applies the combined augmentation techniques
combined_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="img_name",
    y_col="choice",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    preprocessing_function=cold_augmentation.preprocessing_function  # Apply the aggressive augmentation for "cold" class
)

# Use the combined generator for training
model.fit(
    combined_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_df) // batch_size
)

# After defining your generators, you can access class indices like this:
print(train_generator_regular.class_indices)

model.save_weights('scale_down_weights_v4.h5')  # always save your weights after training or during training
model.save('scale_down_model_v4.h5')

# function to classify a single image
#it should return the class of the image (hot, cold, neutral, very hot) as a string
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    proba = model.predict(images, batch_size=10)
    classes = np.argmax(proba, axis=-1)
    # Get the class label from the class index
    class_label = list(train_generator_regular.class_indices.keys())[list(train_generator_regular.class_indices.values()).index(classes)]
    return class_label

def main():
    image_path = '/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/333.jpg'
    print(classify_image(image_path))
    #test accuracy
    score = model.evaluate_generator(validation_generator, len(validation_df) // batch_size)
    print("Accuracy = ", score[1])

if __name__ == '__main__':
    main()
