
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
cmd = "gcc pim_dyn.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim_dyn.so"; os.system(cmd)
cmd = "gcc profile.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o profile.so"; os.system(cmd)

from layers import *
from conv import *
from block import *
from model import *

####

def init_x(num_example, input_shape, xlow, xhigh):
    h, w = input_shape
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    scale = (np.max(x_test) - np.min(x_test)) / (xhigh - xlow)

    x_test = x_test[0:num_example, 0:h, 0:w, :]
    x_test = x_test / scale
    x_test = np.floor(x_test)
    x_test = np.clip(x_test, xlow, xhigh)
    
    x_test = x_test.astype(int)
    
    y_test = y_test[0:num_example].reshape(-1)
    
    return x_test, y_test

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
'wl': 128,
'bl': 128,
'offset': 128,
}

############

arch_params1 = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'sigma': [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
'cards': [1],
'profile': [1],
'rpr_alloc': ['dynamic', 'centroids']
}

arch_params2 = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'sigma': [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
'cards': [0],
'profile': [1],
'rpr_alloc': ['dynamic']
}

############

arch_params = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'sigma': [0.10],
'cards': [1],
'profile': [1],
'rpr_alloc': ['dynamic']
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
    layers = [
    Conv(input_size=(32,32, 3), filter_size=(3,3, 3,64), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    Conv(input_size=(32,32,64), filter_size=(3,3,64,64), pool=2, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),

    Conv(input_size=(16,16,64),  filter_size=(3,3, 64,128), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    Conv(input_size=(16,16,128), filter_size=(3,3,128,128), pool=2, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),

    Conv(input_size=(8,8,128), filter_size=(3,3,128,256), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    Conv(input_size=(8,8,256), filter_size=(3,3,256,256), pool=2, stride=1, pad1=1, pad2=1, params=array_params, weights=weights)
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
x, y = init_x(1, (32, 32), 0, 127)

##########################

weights = np.load('../cifar10_weights.npy', allow_pickle=True).item()
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








