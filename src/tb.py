
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading
import time

cmd = "gcc pim.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim.so"
os.system(cmd)

from layers import Model
from layers import Conv
from layers import Dense
from defines import *

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

def perms(param):
    params = [param]
    
    for key in param.keys():
        val = param[key]
        if type(val) == list:
            new_params = []
            for ii in range(len(val)):
                for jj in range(len(params)):
                    new_param = copy.copy(params[jj])
                    new_param[key] = val[ii]
                    new_params.append(new_param)
                    
            params = new_params
            
    return params

####

param_sweep = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'skip': 1,
'cards': [0, 1],
'stall': 0,
'wl': 128,
'bl': 64,
'offset': 128,
'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
'err_sigma': 0.,
}

param_sweep = perms(param_sweep)

####

weights = np.load('../cifar10_weights.npy', allow_pickle=True).item()

# dont think this padding is right.
layers = [
Conv(input_size=(32,32,3),  filter_size=(3,3,3,32),  stride=1, pad1=1, pad2=1, params=params, weights=weights[0]),
Conv(input_size=(32,32,32), filter_size=(3,3,32,32), stride=2, pad1=1, pad2=1, params=params, weights=weights[1]),

Conv(input_size=(16,16,32), filter_size=(3,3,32,64), stride=1, pad1=1, pad2=1, params=params, weights=weights[2]),
Conv(input_size=(16,16,64), filter_size=(3,3,64,64), stride=2, pad1=1, pad2=1, params=params, weights=weights[3]),

Conv(input_size=(8,8,64), filter_size=(3,3,64,128), stride=1, pad1=1, pad2=1, params=params, weights=weights[4]),
Conv(input_size=(8,8,128), filter_size=(3,3,128,128), stride=2, pad1=1, pad2=1, params=params, weights=weights[5]),
]

model = Model(layers=layers)

####

tests = [
('cnn1', 10, (32, 32), model)
]

####

start = time.time()

for test in tests:
    name, num_example, input_shape, model = test
    x = init_x(num_example, input_shape, 0, 127)
    assert (np.min(x) >= 0 and np.max(x) <= 127)
    _, metrics = model.forward(x=x)
    np.save(name, metrics)

print (time.time() - start)

####








