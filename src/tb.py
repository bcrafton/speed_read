
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading
import time
import copy
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

cmd = "g++ pim.c array.c block.c layer.c layer_sync.c params.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim.so"; os.system(cmd)
cmd = "gcc profile.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o profile.so"; os.system(cmd)

from layers import *
from conv import *
from block import *
from model import *

####

def quantize_np(x):
  scale = 127 / np.max(np.absolute(x))
  x = x * scale
  x = np.round(x)
  x = np.clip(x, -127, 127)
  return x, scale

def init_x(num_example):
    dataset = np.load('../imagenet.npy', allow_pickle=True).item()
    xs, ys = dataset['x'], dataset['y']
    assert (np.shape(xs) == (10, 224, 224, 3))

    # TODO: make sure we are using the right input images and weights
    # xs = xs / 255. 
    # xs = xs - np.array([0.485, 0.456, 0.406])
    # xs = xs / np.array([0.229, 0.224, 0.225])
    # xs, scale = quantize_np(xs)
    
    xs = xs[0:num_example]
    ys = ys[0:num_example]
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

############

array_params = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'adc_mux': 8,
'wl': 256,
'bl': 256,
'offset': 128,
}

############

arch_params1 = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'sigma': [0.05, 0.10],
'cards': [1],
'profile': [1],
'rpr_alloc': ['dynamic', 'centroids']
}

arch_params2 = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'sigma': [0.05, 0.10],
'cards': [0],
'profile': [1],
'rpr_alloc': ['dynamic']
}

############

arch_params = {
'skip': [1],
'alloc': ['block', 'layer'],
'narray': [2 ** 13],
'sigma': [0.10],
'cards': [1],
'profile': [0, 1],
'rpr_alloc': ['centroids', 'dynamic']
}

############

param_sweep = perms(arch_params)

'''
param_sweep1 = perms(arch_params1)
param_sweep2 = perms(arch_params2)
param_sweep = param_sweep1 + param_sweep2
'''

####

def create_model(weights):
    layers=[
    Conv(input_size=(224, 224, 3), filter_size=(7,7,3,64), pool=1, stride=2, pad1=3, pad2=3, params=array_params, weights=weights),
    
    MaxPool(input_size=(112, 112, 64), kernel_size=3, stride=2, params=array_params, weights=weights),
    
    Block1(input_size=(56, 56, 64), filter_size=(64, 64), stride=1, params=array_params, weights=weights),
    Block1(input_size=(56, 56, 64), filter_size=(64, 64), stride=1, params=array_params, weights=weights),
    
    Block2(input_size=(56, 56, 64),  filter_size=(64,  128), stride=2, params=array_params, weights=weights),
    Block1(input_size=(28, 28, 128), filter_size=(128, 128), stride=1, params=array_params, weights=weights),
    
    Block2(input_size=(28, 28, 128), filter_size=(128, 256), stride=2, params=array_params, weights=weights),
    Block1(input_size=(14, 14, 256), filter_size=(256, 256), stride=1, params=array_params, weights=weights),
    
    Block2(input_size=(14, 14, 256), filter_size=(256, 512), stride=2, params=array_params, weights=weights),
    Block1(input_size=(  7, 7, 512), filter_size=(512, 512), stride=1, params=array_params, weights=weights),
    ]

    model = Model(layers=layers)
    return model

####

def run_command(x, y, model, params, return_dict):
    print (params)
    model.init(params)
    if params['profile']:
        model.profile(x=x)
    
    _, result = model.forward(x=x, y=y)
    return_dict[(params['skip'], params['cards'], params['alloc'], params['profile'], params['narray'], params['sigma'], params['rpr_alloc'])] = result

####

results = {}

start = time.time()
x, y = init_x(num_example=1)

##########################

weights = np.load('../resnet18_quant_weights.npy', allow_pickle=True).item()
model = create_model(weights)

##########################

load_profile_adc = False

if not load_profile_adc:
    profile = model.profile_adc(x=x)
    np.save('profile_adc', profile)
else:
    profile = np.load('profile_adc.npy', allow_pickle=True).item()
    model.set_profile_adc(profile)

##########################

num_runs = len(param_sweep)
parallel_runs = 8
for run in range(0, num_runs, parallel_runs):
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for parallel_run in range(min(parallel_runs, num_runs - run)):
        args = (np.copy(x), np.copy(y), copy.copy(model), param_sweep[run + parallel_run], return_dict)
        t = multiprocessing.Process(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    results.update(return_dict)

np.save('results', results)
print ('time taken:', time.time() - start)

####








