import numpy as np
import tensorflow as tf
import keras
import math

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
NUM_IMG = 10000
SIZE = 28
BATCHES = 100
ALPHA = 0.1

print(x_train[0])
print(y_train[0])

def init():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def flatten(arr):
    output = []
    for a in arr:
        for item in a:
            output.append(item)
    return output

def activation(arr):
    for n in range(len(arr)):
        arr[n] = 1/(1+math.exp(-1*arr[n]))
    return arr

def cost():
    pass

def forward_prop(x, W1, b1, W2, b2):
    x1 = []
    for i in range(10):
        x1.append(activation(sum(np.add(np.multiply(x,W1[i]),b1[i]))))
    x2 = []
    for i in range(10):
        x2.append(activation(sum(np.add(np.multiply(x1,W2[i]),b2[i]))))
    return x1, x2

def back_prop():
    pass

def train():
    for n in range(0, NUM_IMG-BATCHES, BATCHES):
        W1, b1, W2, b2 = init()
        for i in range(n, n+BATCHES):
            x = flatten(x_train[i])
            x1, x2 = forward_prop(x,W1,b1,W2,b2)
            cost = 0
            for j in range(10):
                if j == y_train[i]:
                    cost += (1 - x2[j])**2
                else:
                    cost += (x2[j])**2
            cost /= 20
