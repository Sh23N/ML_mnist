from tensorflow import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

import tensorflow as tf

model=keras.Sequential()
model.add(keras.layers.Conv2D(32, (3,3),strides=(1,1),padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Conv2D(64, (3,3),strides=(1,1),padding='valid', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Conv2D(128, (3,3),strides=(1,1),padding='valid', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model1=keras.Sequential()
model1.add(keras.layers.Conv2D(32, (3,3),strides=(1,1),padding='valid', input_shape=(28,28,1)))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Activation('relu'))
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model1.add(keras.layers.Conv2D(32, (3,3),strides=(1,1),padding='valid', input_shape=(28,28,1)))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Activation('relu'))
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dropout(0.2))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history1=model1.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),batch_size=256)

history=model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),batch_size=256)

import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'],color='blue')
plt.plot(model.history.history['val_accuracy'],color='red')

import matplotlib.pyplot as plt

plt.plot(model1.history1.history['accuracy'],color='blue')
plt.plot(model1.history1.history['val_accuracy'],color='red')



