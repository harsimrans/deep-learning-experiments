#!/usr/bin/env python

from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers

import numpy as np
import time


IMG_HEIGHT = 150
IMG_WIDTH = 150
NB_VALIDATION_SAMPLES = 800
NB_TRAIN_SAMPLES = 2000
BATCH_SIZE = 16
EPOCHS = 50


# this section trains a NN from scratch. [achieves roughly 80% accuracy]

print("---------------------------------------")
print("Training the classifier from scratch...")
print("---------------------------------------")

tic = time.time()

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  
        target_size=(IMG_HEIGHT, IMG_WIDTH),  
        batch_size=BATCH_SIZE,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(IMG_HEIGHT, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps= NB_VALIDATION_SAMPLES // BATCH_SIZE)

toc = time.time()
model.save_weights('simple_model_from_scratch.h5')  

print("Time taken: ", (toc-tic), "s")


#### this sections trains the dense layers on bottleneck features from VGG-16. [roughly 90% accuracy]

print("---------------------------------------")
print("       Build on top of VGG-16...       ")
print("---------------------------------------")

tic = time.time()

# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,  
        shuffle=False)  

bottleneck_features_train = model.predict_generator(generator, NB_TRAIN_SAMPLES//BATCH_SIZE)
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

generator = datagen.flow_from_directory(
        'data/validation',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

bottleneck_features_validation = model.predict_generator(generator, NB_VALIDATION_SAMPLES//BATCH_SIZE)
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

train_data = np.load(open('bottleneck_features_train.npy'))
train_labels = np.array([0] * (NB_TRAIN_SAMPLES/2) + [1] * (NB_TRAIN_SAMPLES/2))

validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.array([0] * (NB_VALIDATION_SAMPLES/2) + [1] * (NB_VALIDATION_SAMPLES/2))

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=EPOCHS,
          BATCH_SIZE=BATCH_SIZE,
          validation_data=(validation_data, validation_labels))

toc = time.time()
model.save_weights('bottleneck_fc_model.h5')

print("Time taken: ", (toc-tic), "s")


### this section corresponds to fine tuning of model [roughly 95% accuracy]

print("--------------------------------------------------")
print ("fine tune the model learnt with VGG-16 as base...")
print("--------------------------------------------------")

tic = time.time()

top_model_weights_path = "./bottleneck_fc_model.h5"
base_model = applications.VGG16(
				include_top=False, 
				weights='imagenet', 
				input_shape=(150, 150, 3))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights(top_model_weights_path)

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

for layer in model.layers[:13]:
    layer.trainable = False

print (model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

# fine-tune the model
model.fit_generator(
        train_generator,
        steps_per_EPOCH=NB_TRAIN_SAMPLES // BATCH_SIZE,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE)

toc = time.time()
model.save_weights('fine_tuned_model.h5')
print("Time taken: ", (toc-tic), "s")