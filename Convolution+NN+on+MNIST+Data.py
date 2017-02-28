
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.utils import np_utils


# In[2]:

ds = pd.read_csv('./train.csv')
data = ds.values


# In[3]:

split = int(0.75 * data.shape[0])
X_train = data[:split, 1:]/255.0
X_val = data[split:, 1:]/255.0

y_train = np_utils.to_categorical(data[:split, 0])
y_val = np_utils.to_categorical(data[split:, 0])

X_train = X_train.reshape((split, 1, 28, 28))
X_val = X_val.reshape((data.shape[0]-split, 1, 28, 28))

print X_train.shape, y_train.shape
print X_val.shape, y_val.shape


# In[5]:

# Model creation

model = Sequential()
model.add(Convolution2D(16, 5, 5, input_shape=(1,28,28)))    #1,28,28 for Theano; 28,28,1 for TensorFlow

# Taking 16 Kernels; of size 5x5. Therefore, resultant matrix of (28-5+1)*(28-5+1) size. But took 16 kernels, so a tensor of 16x24x24
model.add(MaxPooling2D(pool_size = (2,2)))

# Pooled each of the resultant 24*24 matrix into 2x2 grids, giving 16x12x12 tensor
model.add(Activation('relu'))

model.add(Convolution2D(8,3,3))

# Taking 8 Kernels; of size 3x3. Therefore, resultant matrix of (24-3+1)*(24-3+1) size. But took 8 kernels, so a tensor of 8x21x21
model.add(MaxPooling2D(pool_size=(3,3)))

# Pooled each of the resultant 21*21 matrix into 3x3 grids, giving 16x12x12 tensor. (3x3 makes covers the entire image)
model.add(Activation('relu'))

model.add(Flatten())           # Flattening the tensors
model.add(Dense(10))           # Last layer - For final classification. 10 neurons in this layer.
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

hist = model.fit(X_train, y_train, nb_epoch = 25, shuffle=True, batch_size = 100, validation_data = (X_val, y_val))


# In[ ]:

# Result analysis

plt.figure(0)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.show()

