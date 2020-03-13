
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading
import time

from dot import *

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

X = 64
K = 3
C = 64
N = 64

xs = np.random.randint(low=0, high=2, size=(1,X,X,N)).astype(int)
f = np.random.randint(low=0, high=2, size=(K,K,C,N)).astype(int)
b = np.zeros(shape=N)
q = np.ones(shape=N)

####

start = time.time()
for x in xs:
    y1 = conv_ref(x, f, b, q, 1, 1, 1)
print (time.time() - start)

####

start = time.time()
for x in xs:
    y2 = conv_ref2(x, f, b, q, 1, 1, 1)
print (time.time() - start)

####

print (np.all(y1 == y2))

'''
print (x[:, :, 0])
print (y1[:, :, 0])
print (y2[:, :, 0])

print (np.count_nonzero(y1) / np.prod(np.shape(y1)))
print (np.count_nonzero(y2) / np.prod(np.shape(y2)))
'''

####





