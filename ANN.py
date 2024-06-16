import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist=keras.datasets.mnist
(train_features,train_labels),(test_features,test_labels)=mnist.load_data()

train_features.shape

test_features.shape

img=train_features[1000]
print(train_labels[1000])
plt.imshow(img)

np.min(img),np.max(img)

train_features=train_features/255.0
test_features=test_features/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax') # [0.25  0.25 0.5] to=> [0 0 1] sample is bilong to thirt=d class
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

hist=model.fit(train_features,train_labels,epochs=100,batch_size=256,validation_data=(test_features,test_labels))

indx=600
img=test_features[indx]
plt.imshow(img)
print(test_labels[indx])
model.predict(np.array([img]))

acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
plt.plot(acc,label='accuracy',color='red')
plt.plot(val_acc,label='val_accuracy',color='blue')
plt.title('model accuracy')

plt.show()

