import os
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# Define the image data generator for augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# Define the batch size, target image size, and number of classes
batch_size = 32
target_size = (24, 24)
num_classes = 2


# Define the train and validation generators
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\yash1\OneDrive\Desktop\exp\dataset_new\train',
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\yash1\OneDrive\Desktop\exp\dataset_new\test',
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')


# Define the model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)


# Save the model
model.save('models/cnnCat2.h5', overwrite=True)
