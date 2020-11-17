
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

from tests import CC
from tests import BB
from tests import Thresh
from tests import CE
from tests import Simple
from tests import dac2

############

array_params, arch_params = CC()
# array_params, arch_params = BB()
# array_params, arch_params = Thresh()
# array_params, arch_params = CE()
# array_params, arch_params = Simple()
# array_params, arch_params = dac2()

############

def run_command(x, y, model, params, return_list):
    print (params)
    
    model.init(params)
    if params['profile']:
        model.profile(x=x)
    
    out, out_ref, result = model.forward(x=x, y=y)
    abs_error = np.mean(np.absolute(out - out_ref))
    abs_mean = np.mean(np.absolute(out_ref))

    # return_dict[(params['skip'], params['cards'], params['alloc'], params['profile'], params['narray'], params['sigma'], params['rpr_alloc'])] = result

    update = {
    'skip':      params['skip'],
    'cards':     params['cards'],
    'alloc':     params['alloc'],
    'profile':   params['profile'],
    'narray':    params['narray'],
    'sigma':     params['sigma'],
    'rpr_alloc': params['rpr_alloc'],
    'thresh':    params['thresh'],
    'abs_error': abs_error
    }
    
    for r in result:
        r.update(update)
        return_list.append(r)
        
####

model, x, y = load_resnet(num_example=1, array_params=array_params)
# model, x, y = load_cifar(num_example=2, array_params=array_params)

####

start = time.time()

load_profile_adc = True

if not load_profile_adc:
    profile = model.profile_adc(x=x)
    np.save('profile_adc', profile)
else:
    profile = np.load('profile_adc.npy', allow_pickle=True).item()

model.set_profile_adc(profile)

##########################

num_runs = len(arch_params)
parallel_runs = 8

thread_results = []
manager = multiprocessing.Manager()
for _ in range(num_runs):
    thread_results.append(manager.list())

for run in range(0, num_runs, parallel_runs):
    threads = []
    
    for parallel_run in range(min(parallel_runs, num_runs - run)):
        args = (np.copy(x), np.copy(y), copy.copy(model), arch_params[run + parallel_run], thread_results[run + parallel_run])
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








