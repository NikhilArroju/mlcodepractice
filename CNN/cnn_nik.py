# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 15:42:49 2018

@author: Nik
"""

from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten

classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation = 'relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPool2D(pool_size=(3,3)))

classifier.add(Flatten())

classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dense(units = 1,activation = 'sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1./255)

training_set = train_data_gen.flow_from_directory('dataset/training_set',target_size = (128, 128),
                                                 batch_size = 32,class_mode = 'binary')

test_set = test_data_gen.flow_from_directory('dataset/test_set',target_size = (128, 128),
                                                 batch_size = 32,class_mode = 'binary')

classifier.fit_generator(training_set,steps_per_epoch=200,epochs = 10,validation_data=test_set,
                         validation_steps=(2000/32))


