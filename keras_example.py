# Thanks to:
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# for the guide.

import numpy as np
# initial seed for reproducibility
np.random.seed(3411)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Load the training and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from matplotlib import pyplot as plt

# Uncomment to see a visual representation of
# the input data
plt.imshow(X_train[1])

# We have to include the depth of the image.
# The depth is the number of channels. HSV or RGB
# have three channels. These images have only one
# channel (black and white).
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# This is a rather manual way to normalize the
# data. But, we want values between 0 and 1, not 0
# and 255.
X_train /= 255
X_test /= 255

# Normally the training data (expected outputs) is
# just a single number per input object. However,
# to compare the expected vs calculated values in
# back-propagation, you need to have the Y values
# be an 10 length array. 
# ie: 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# Are we training a model from file, or creating 
# a new one?
model = None
current_model_path = "models/model2"

try:
    model = keras.models.load_model(current_model_path)
except IOError:
    model = Sequential()

    # Set the model architecture:
    # softmax in the last layer + 
    # cross-entropy loss function is good for 
    # categorization
    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(28, 28, 1), data_format="channels_last"))
    # uncomment the next line to make the
    # performance worse 
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

# Do the learning!
# The train data is split into 70% for training
# and 30% for validation.
model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.3,  verbose=1)

model.save(current_model_path)

# How did we do?
score = model.evaluate(X_test, Y_test, verbose=0)
print(f'Model sore: {score}')
