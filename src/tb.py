
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

############

# NOTE: This has to come before any module that loads the c code
# before, this came after 'load_resnet', 'load_cifar' and so it loaded the previous version each time.

cmd = "g++ pim.c array.c block.c layer.c layer_sync.c params.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o pim.so"
ret = os.system(cmd)
assert (ret == 0)

cmd = "gcc profile.c -DPYTHON_EXECUTABLE=/usr/bin/python3 -fPIC -shared -o profile.so"
ret = os.system(cmd)
assert (ret == 0)

############

from resnet import load_resnet
from cifar import load_cifar

############

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
'max_rpr': 64,
}

############

arch_params = {
'skip': [1],
'alloc': ['block'],
'narray': [2 ** 13],
'cards': [1],
'profile': [1],
'rpr_alloc': ['static'],
'sigma': [0.04],
'thresh': [1.00]
}

############

param_sweep = perms(arch_params)

####

def run_command(x, y, model, params, return_list):
    # print (params)
    
    model.init(params)
    if params['profile']:
        model.profile(x=x)
    
    _, result = model.forward(x=x, y=y)
    
    # return_dict[(params['skip'], params['cards'], params['alloc'], params['profile'], params['narray'], params['sigma'], params['rpr_alloc'])] = result
    
    update = {
    'skip':      params['skip'],
    'cards':     params['cards'],
    'alloc':     params['alloc'],
    'profile':   params['profile'],
    'narray':    params['narray'],
    'sigma':     params['sigma'],
    'rpr_alloc': params['rpr_alloc'],
    'thresh':    params['thresh']
    }
    
    for r in result:
        r.update(update)
        return_list.append(r)
        
####

# model, x, y = load_resnet(num_example=1, array_params=array_params)
model, x, y = load_cifar(num_example=1, array_params=array_params)

####

start = time.time()

load_profile_adc = False

if not load_profile_adc:
    profile = model.profile_adc(x=x)
    np.save('profile_adc', profile)
else:
    profile = np.load('profile_adc.npy', allow_pickle=True).item()

model.set_profile_adc(profile)

##########################

num_runs = len(param_sweep)
parallel_runs = 4

thread_results = []
manager = multiprocessing.Manager()
for _ in range(num_runs):
    thread_results.append(manager.list())

for run in range(0, num_runs, parallel_runs):
    threads = []
    
    for parallel_run in range(min(parallel_runs, num_runs - run)):
        args = (np.copy(x), np.copy(y), copy.copy(model), param_sweep[run + parallel_run], thread_results[run + parallel_run])
        t = multiprocessing.Process(target=run_command, args=args)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

results = []
for r in thread_results:
    results.extend(r)

np.save('results', results)
print ('time taken:', time.time() - start)

####








