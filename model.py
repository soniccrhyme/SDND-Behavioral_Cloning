#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:03:42 2017

model.py

script used to create and train the model

@author: ucalegon
"""

import csv
import numpy as np
from skimage.io import imread
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D

path = 'data/'
BATCH_SIZE = 196

samples = []

with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

# Generator to save overutilization of RAM
def generator(samples, batch_size = BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                img_name = path+'IMG/'+sample[0].split('/')[-1]
                l_img_name = path+'IMG/'+sample[1].split('/')[-1]
                r_img_name = path+'IMG/'+sample[2].split('/')[-1]

                # Add center image and respective angle
                center_image = imread(img_name)
                center_angle = float(sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Add flipped image and reversed angle
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                # Add Left and Right images with respective angle
                # Modify angle for right and left image by offset
                offset = 0.2

                l_image = imread(l_img_name)
                l_angle = float(sample[3])+offset
                images.append(l_image)
                angles.append(l_angle)

                r_image = imread(r_img_name)
                r_angle = float(sample[3])-offset
                images.append(r_image)
                angles.append(r_angle)

                # Add flipped l/r image and reversed angle
                l_image_flipped = np.fliplr(l_image)
                l_angle_flipped = -l_angle
                images.append(l_image_flipped)
                angles.append(l_angle_flipped)

                r_image_flipped = np.fliplr(r_image)
                r_angle_flipped = -r_angle
                images.append(r_image_flipped)
                angles.append(r_angle_flipped)

            X = np.array(images)
            y = np.array(angles)
            yield shuffle(X,y)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Pipeline, based on NVIDIA's end to end learning paper
model = Sequential()
model.add(Cropping2D(cropping = ((70,25),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: (x-128.)/128.))
model.add(Conv2D(filters = 24, kernel_size = 5, strides = (2,2), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 36, kernel_size = 5, strides = (2,2), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 48, kernel_size = 5, strides = (2,2), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(rate = 0.75))
model.add(Dense(50))
model.add(Dropout(rate = 0.75))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, steps_per_epoch = int(len(train_samples)/BATCH_SIZE), epochs = 10, validation_data = validation_generator, validation_steps = int(len(validation_samples)/BATCH_SIZE))

model.save('model.h5')
