import numpy as np
# initial seed
# np.random.seed(123)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from matplotlib import pyplot as plt
plt.imshow(X_train[1])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = None
current_model_path = "models/model1"

try:
    model = keras.models.load_model(current_model_path)
except IOError:
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(28, 28, 1), data_format="channels_last"))
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

model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

model.save(current_model_path)

score = model.evaluate(X_test, Y_test, verbose=0)
print(f'Model sore: {score}')