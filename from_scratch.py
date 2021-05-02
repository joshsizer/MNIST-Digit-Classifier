"""
Created on Sun April 25 2021

Copyright (c) 2021 - Joshua Sizer

This code is licensed under MIT license (see
LICENSE for details)
"""

import pandas as pd
import numpy as np
import numpy.random as nprand
nprand.seed(3321)
import math

def full_print(arr):
    """Disable the corners only functionality for
    this specific print job.
    """
    np.set_printoptions(threshold=np.inf)
    print(arr)
    np.set_printoptions(threshold=1000)

def to_categorical(Y):
    """Convert a single number to a vector, where
    the value of 1 is set for the index equal to the
    single number, and 0 otherwise. 
    """
    new_arr = np.zeros((len(Y), 10), Y.dtype)
    for i in range(len(Y)):
        new_arr[i][Y[i][0]] = 1

    return new_arr

def from_categorical(Y):
    """Turn a categorical output into a single
    digit classification.
    """
    new_arr = np.zeros((len(Y), 1), Y.dtype)
    for i in range(len(Y)):
        new_arr[i][0] = np.argmax(Y[i])

    return new_arr

def sigmoid(x):
    """An implementation of sigmoid function.

    Thanks to source:
    https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Avoid overflow by limiting the range of X
    x = np.clip(x, -500, 500)
    sig_x = sigmoid(x)
    return sig_x * (1-sig_x)

def relu(x):
    """An implementation of a ReLU (rectified
    linear unit) function.

    Thanks to source: 
    https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    """
    return np.maximum(x, 0)


def relu_derivative(x):
    """An implementation of the ReLU derivative.

    Thanks to source:
    https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
    """
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(x):
    """An implementation of the softmax function

    Thanks to source:
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(x):
    pass


def initialize_layer_weights(n, m, init_type="xavier"):
    """Generates a distribution of random numbers in some range,
    specified by the init_type parameter.
    n is the number of input nodes 
    m is the number of output nodes.
    U is the uniform distribution. 
    G is the gaussian or normal distribution.

    Thanks to source: 
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

    Initialization type (init_type) can be:

    Good for Sigmoid and Tanh activation functions.
    "xavier"
    Xavier Glorot uses the formula: 
        weight = U[-(1/sqrt(n)), 1/sqrt(n)]

    "nxavier"
    Normalized Xavier Glorot uses the formula:
        weight = U[-(sqrt(6)/sqrt(n+m)), sqrt(6)/sqrt(n+m)]

    Good for ReLU activation funtions.
    "he"
    Kaiming He uses the formula:
        weight = G(0.0, sqrt(2/n))
    """
    if "xavier" in init_type:
        numbers = nprand.rand(m, n)
        if init_type == "xavier":
            lower, upper = -(1.0 / math.sqrt(n)), (1.0 / math.sqrt(n))
        else:
            lower, upper = -(math.sqrt(6.0) / math.sqrt(n + m)), (math.sqrt(6.0) / math.sqrt(n + m))
        scaled = lower + numbers * (upper - lower)
    else:
        std = math.sqrt(2.0 / n)
        numbers = nprand.randn(m, n)
        scaled = numbers * std

    return scaled

def shuffle_two(a, b):
    """Shuffle two arrays in the same way so as to
    keep them correctly aligned with each other.

    For example:
    shuffle_two([1, 2, 3], [3, 2, 1])
    could produce [1, 3, 2], [3, 1, 2]
    """
    rnd_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rnd_state)
    np.random.shuffle(b)

# Load train.csv 
df = pd.read_csv("data/train.csv")
dfn = df.to_numpy()

# Split into X and Y
dfn = np.hsplit(dfn, [1]) 

# X are the input objects (MNIST digits) and Y are
# the correct label for each object.
X_train = dfn[1]
Y_train = dfn[0]

# Normalize our inputs and outputs
X_train = X_train / 255
Y_train = to_categorical(Y_train)

# Define our layers
layers = [X_train.shape[1], 128, 10]

# Initialize our layer 1 and layer 2 weights. Layer
# 1 uses a ReLU function, so use "he" init type.
# Layer 2 uses a sigmoid function, so use the
# xavier or nxavier type.
l1_w = initialize_layer_weights(layers[0], layers[1], "he")
l2_w = initialize_layer_weights(layers[1], layers[2], "nxavier")

# Initialize our bias to 0 for all layers
l1_b = np.zeros((layers[1], 1))
l2_b = np.zeros((layers[2], 1))

X_total = X_train
Y_total = Y_train
X_validate = None
Y_validate = None

# Do the learning
mini_batch_size = 32
epochs = 10

for i in range(epochs):
    # Shuffle our dataset
    shuffle_two(X_total, Y_total)

    # Split our dataset into validation and train
    # sets.
    split_point = int(len(X_total) * 0.3)
    
    X_validate = X_total[0:split_point]
    X_train = X_total[split_point:]
    Y_validate = Y_total[0:split_point]
    Y_train = Y_total[split_point:]

    for i in range(0, len(X_train), mini_batch_size):
        mini_batch_x = X_train[i: i + mini_batch_size].T
        mini_batch_y = Y_train[i: i + mini_batch_size].T

        z1 = l1_w.dot(mini_batch_x) + l1_b
        a1 = relu(z1)
        z2 = l2_w.dot(a1) + l2_b
        a2 = sigmoid(z2)

        cost = np.mean(np.square(a2 - mini_batch_y))
        #print(cost)

        da2 = (a2 - mini_batch_y)
        dz2 = da2 * sigmoid_derivative(z2)
        nabla_l2_b = (1/mini_batch_size) * np.sum(dz2, axis=1, keepdims=True)
        nabla_l2_w = (1/mini_batch_size) * np.dot(dz2, a1.T)

        da1 = np.dot(l2_w.T, dz2)
        dz1 = da1 * relu_derivative(a1)
        nabla_l1_b = (1/mini_batch_size) * np.sum(dz1, axis=1, keepdims=True)
        nabla_l1_w = (1/mini_batch_size) * np.dot(dz1, mini_batch_x.T)

        learning_rate = 0.2

        l1_w = l1_w - learning_rate * nabla_l1_w
        l2_w = l2_w - learning_rate * nabla_l2_w
        l1_b = l1_b - learning_rate * nabla_l1_b
        l2_b = l2_b - learning_rate * nabla_l2_b

# check out accuracy
y = from_categorical(Y_validate)
x = X_validate

z1 = l1_w.dot(x.T) + l1_b
a1 = relu(z1)
z2 = l2_w.dot(a1) + l2_b
a2 = sigmoid(z2)

predictions = from_categorical(a2.T)

count_correct = 0
total = 0

for i in range(len(x)):
    prediction = predictions[i]
    expected = y[i][0]
    if prediction == expected:
        count_correct += 1
    total += 1

print(count_correct/total)



    


