import numpy as np
import tensorflow as tf
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
NUMIMG = 10000
SIZE = 28

