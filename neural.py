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
        temp = []
        for item in a:
            temp.append(item)
        output.append(temp)
    return output

def activation(arr):
    for n in range(len(arr)):
        arr[n] = 1/(1+math.exp(-1*arr[n]))
    return arr

def cost():
    pass

def forward_prop(x, W1, b1, W2, b2):
    #combine the W's and b's into matrices
    #to allow easy looping.
    #take previous x and np.multiply with W[n]
    #then np.add b[n]
    #add these together for each node then put
    #through sigmoid activation
    #add result to x and repeat
    #return x
    for n in range(len(W1)):
        pass

def back_prop():
    pass

def train():
    for n in range(0, NUM_IMG-BATCHES, BATCHES):
        W1, b1, W2, b2 = init()
        for i in range(n, n+BATCHES):
            x = [flatten(x_train[i])]
            #get x from forward
            #calculate cost
            #do backprop
