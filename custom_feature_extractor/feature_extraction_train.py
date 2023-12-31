import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256

# Define the ImageDataGenerator for training data
train_datagen = ImageDataGenerator(validation_split=0.1)  # Set validation_split to 0.1 for 10% validation data

# Define the path to your training data
train_data_path = 'vehicle_dataset_train'

# Create the training generator
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse",
    subset="training"  # Specify that this is the training subset
)

# Create the validation generator
validation_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse",
    subset="validation"  # Specify that this is the validation subset
)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    'vehicle_dataset_test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse"
)

sz = 128

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))  # Additional convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))  # More units in the dense layer
model.add(Dropout(0.5))  # Increased dropout rate

model.add(Dense(units=256, activation='relu'))  # Additional dense layer
model.add(Dropout(0.4))

model.add(Dense(units=184, activation='softmax'))

# This model has a total of 11 layers including 4 Conv2D, 4 MaxPooling2D, and 3 Dense layers.

print(model.summary())  

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#model = tf.keras.models.load_model('vehicle_train')
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

tf.saved_model.save(model, 'custom_feature_model')