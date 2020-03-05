
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

cnn1_layers = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(2,2,32,32), stride=2, pad1=0, pad2=0),
]
cnn1 = model(layers=cnn1_layers)

####

tests = {
'cnn1': (3, (8, 8), cnn1)
}

####

for key in tests.keys():
    num_example, input_shape, model = tests[key]
    x = init_x(num_example, input_shape, 0, 127)
    model.forward(x=x)

####








