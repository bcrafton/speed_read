
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading
import time
import copy

import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

cmd = "gcc pim.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim.so"
os.system(cmd)

from layers import Model
from layers import Conv
from defines import *

####

def init_x(num_example, input_shape, xlow, xhigh):
    h, w = input_shape
    # (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_test = np.load('../imagenet.npy').item()['data']
    x_test = np.reshape(x_test, (100, 64, 64, 3))
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
'skip': [0, 1],
'cards': [0, 1],
'stall': 0,
'wl': 128,
'bl': 128,
'offset': 128,
'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
'err_sigma': 0.,
}

param_sweep = perms(param_sweep)

####

def create_model(weights, params):
    # dont think this padding is right.
    layers = [
    Conv(input_size=(64,64,3),   filter_size=(3,3,3,64),     stride=1, pad1=1, pad2=1, params=params, weights=weights[0]),
    Conv(input_size=(64,64,64),  filter_size=(3,3,64,64),    stride=2, pad1=1, pad2=1, params=params, weights=weights[1]),
    Conv(input_size=(32,32,64),  filter_size=(3,3,64,128),   stride=1, pad1=1, pad2=1, params=params, weights=weights[2]),
    Conv(input_size=(16,16,128), filter_size=(3,3,128,256),  stride=2, pad1=1, pad2=1, params=params, weights=weights[3]),
    Conv(input_size=(8,8,256),   filter_size=(3,3,256,512),  stride=2, pad1=1, pad2=1, params=params, weights=weights[4]),
    Conv(input_size=(4,4,512),   filter_size=(3,3,512,1024), stride=2, pad1=1, pad2=1, params=params, weights=weights[5]),
    ]

    model = Model(layers=layers)
    return model

####

def run_command(x, weights, params, return_dict):
    print (params)
    model = create_model(weights, params)
    _, result = model.forward(x=x)
    return_dict[(params['skip'], params['cards'], params['sigma'])] = result

####

results = {}

start = time.time()
x = init_x(3, (64, 64), 0, 127)
weights = np.load('../imagenet_weights.npy', allow_pickle=True).item()

num_runs = len(param_sweep)
parallel_runs = 12
for run in range(0, num_runs, parallel_runs):
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for parallel_run in range(min(parallel_runs, num_runs - run)):
        args = (np.copy(x), copy.copy(weights), param_sweep[run + parallel_run], return_dict)
        t = multiprocessing.Process(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    results.update(return_dict)

np.save('results', results)
print (time.time() - start)

####








