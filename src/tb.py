
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading

from layers import Conv
from layers import Dense
from defines import *
from model import model

####

def init_x(num_example, input_shape, xlow, xhigh):
    h, w = input_shape
    (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test[0:num_example, 0:h, 0:w, :]
    
    scale = (np.max(x_test) - np.min(x_test)) / (xhigh - xlow)
    x_test = x_test / scale
    x_test = np.floor(x_test)
    x_test = np.clip(x_test, xlow, xhigh)
    
    x_test = x_test.astype(int)
    return x_test

####

params = {
'bpa': 8,
'bpw': 8,
'rpr': 16,
'adc': 16,
'skip': 1
}

weights = np.load('../cifar10_weights.npy', allow_pickle=True).item()

# TODO - its looking like we want a pooling operation.
layers = [
Conv(input_size=(32,32,3),  filter_size=(3,3,3,32), stride=2, pad1=0, pad2=1, params=params, weights=weights[0]),
Conv(input_size=(16,16,32), filter_size=(3,3,32,64), stride=2, pad1=0, pad2=1, params=params, weights=weights[1]),
Conv(input_size=(8,8,64),   filter_size=(3,3,64,128), stride=2, pad1=0, pad2=1, params=params, weights=weights[2]),
]

# TODO: these have the same name ...
model = model(layers=layers)

####

tests = [
(3, (32, 32), model)
]

####

for test in tests:
    num_example, input_shape, model = test
    x = init_x(num_example, input_shape, 0, 127)
    _, psum = model.forward(x=x)
    print (psum)

####








