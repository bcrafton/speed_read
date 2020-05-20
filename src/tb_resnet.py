
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

cmd = "gcc pim.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim.so"; os.system(cmd)
cmd = "gcc pim_sync.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim_sync.so"; os.system(cmd)

from layers import *
from defines import *

####

def quantize_np(x):
  scale = 127 / np.max(np.absolute(x))
  x = x * scale
  x = np.round(x)
  x = np.clip(x, -127, 127)
  return x, scale

def init_x(num_example):
    dataset = np.load('resnet18_activations.npy', allow_pickle=True).item()
    xs, ys = dataset['x'], dataset['y']
    assert (np.shape(xs) == (10, 224, 224, 3))

    # TODO: make sure we are using the right input images and weights
    # xs = xs / 255. 
    # xs = xs - np.array([0.485, 0.456, 0.406])
    # xs = xs / np.array([0.229, 0.224, 0.225])
    # xs, scale = quantize_np(xs)
    
    xs = xs[1:2]
    ys = ys[1:2]
    return xs, ys

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
'adc_mux': 8,
'skip': [1],
'cards': [0, 1],
'alloc': ['block'],
# 'profile': [0, 1],
'stall': 0,
'wl': 128,
'bl': 128,
'offset': 128,
# 'narray': [2 ** 14, 24960, 2 ** 15],
'narray': [1.5 * 2 ** 14],
# 'narray': [5472],
# seems like you gotta change e_mu based on this.
# set e_mu = 0.15
# set sigma = 0.05
'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
'err_sigma': 0.,

'profile': [1],
}

param_sweep = perms(param_sweep)

####

def create_model(weights, params):
    '''
    layers=[
    conv_block((7,7,3,64), 2, noise=None, weights=weights),
    
    max_pool(2, 3),

    res_block1(64,   64, 1, noise=None, weights=weights),
    res_block1(64,   64, 1, noise=None, weights=weights),

    res_block2(64,   128, 2, noise=None, weights=weights),
    res_block1(128,  128, 1, noise=None, weights=weights),

    res_block2(128,  256, 2, noise=None, weights=weights),
    res_block1(256,  256, 1, noise=None, weights=weights),

    res_block2(256,  512, 2, noise=None, weights=weights),
    res_block1(512,  512, 1, noise=None, weights=weights),

    avg_pool(7, 7),
    dense_block(512, 1000, noise=None, weights=weights)
    ]
    '''
    '''
    layers = [
    Conv(input_size=(32,32, 3), filter_size=(3,3, 3,64), pool=1, stride=1, pad1=1, pad2=1, params=params, weights=weights[0]),
    Conv(input_size=(32,32,64), filter_size=(3,3,64,64), pool=2, stride=1, pad1=1, pad2=1, params=params, weights=weights[1]),

    Conv(input_size=(16,16,64),  filter_size=(3,3, 64,128), pool=1, stride=1, pad1=1, pad2=1, params=params, weights=weights[2]),
    Conv(input_size=(16,16,128), filter_size=(3,3,128,128), pool=2, stride=1, pad1=1, pad2=1, params=params, weights=weights[3]),

    Conv(input_size=(8,8,128), filter_size=(3,3,128,256), pool=1, stride=1, pad1=1, pad2=1, params=params, weights=weights[4]),
    Conv(input_size=(8,8,256), filter_size=(3,3,256,256), pool=2, stride=1, pad1=1, pad2=1, params=params, weights=weights[5])
    ]
    '''
    layers=[
    Conv(input_size=(224, 224, 3), filter_size=(7,7,3,64), pool=1, stride=2, pad1=3, pad2=3, params=params, weights=weights),
    
    MaxPool(input_size=(112, 112, 64), kernel_size=3, stride=2, params=params, weights=weights),
    
    Block1(input_size=(56, 56, 64), filter_size=(64, 64), stride=1, params=params, weights=weights),
    Block1(input_size=(56, 56, 64), filter_size=(64, 64), stride=1, params=params, weights=weights),
    
    Block2(input_size=(56, 56, 64),  filter_size=(64,  128), stride=2, params=params, weights=weights),
    Block1(input_size=(28, 28, 128), filter_size=(128, 128), stride=1, params=params, weights=weights),
    
    Block2(input_size=(28, 28, 128), filter_size=(128, 256), stride=2, params=params, weights=weights),
    Block1(input_size=(14, 14, 256), filter_size=(256, 256), stride=1, params=params, weights=weights),
    
    Block2(input_size=(14, 14, 256), filter_size=(256, 512), stride=2, params=params, weights=weights),
    Block1(input_size=(  7, 7, 512), filter_size=(512, 512), stride=1, params=params, weights=weights),
    
    AvgPool(input_size=(7, 7, 512), kernel_size=7, stride=7, params=params, weights=weights),
    ]

    model = Model(layers=layers, params=params)
    return model

####

def run_command(x, y, weights, params, return_dict):
    print (params)
    model = create_model(weights, params)
    if params['profile']:
        model.profile(x=x)
    _, result = model.forward(x=x, y=y)
    return_dict[(params['skip'], params['cards'], params['alloc'], params['profile'], params['narray'], params['sigma'])] = result

####

results = {}

start = time.time()
x, y = init_x(num_example=1)
# TODO: make sure we are using the right input images and weights
weights = np.load('resnet18_quant_weights.npy', allow_pickle=True).item()

num_runs = len(param_sweep)
parallel_runs = 8
for run in range(0, num_runs, parallel_runs):
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for parallel_run in range(min(parallel_runs, num_runs - run)):
        args = (np.copy(x), np.copy(y), copy.copy(weights), param_sweep[run + parallel_run], return_dict)
        t = multiprocessing.Process(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    results.update(return_dict)

np.save('results', results)
print ('time taken:', time.time() - start)

####







