
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compiler', type=str, default='iverilog')
args = parser.parse_args()

import numpy as np
import tensorflow as tf
import threading

from layers import Conv
from layers import Dense

from compiler import compile_code
from compiler import compile_pcm
from compiler import compile_emu
from compiler import total_macs
from compiler import total_pcm_scans
from compiler import total_aram_scans

from emu_block import emu

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
    
# you wont get this perfectly correct because of zero skipping.
def calc_runtime(clk_period, input_shape, model):
    h, w = input_shape
    write_cycle = h * w 
    
    mac = total_macs(model)
    mac_per_clk = 256
    mac_cycle = mac / mac_per_clk
    mac_runtime = clk_period * mac_cycle 

    pcm_scan = 0 # total_pcm_scans(model)
    aram_scan = total_aram_scans(model) // 2 # (2) 16 bit aram value per write
    x_scan = np.prod(input_shape)
    icache_scan = len(model) * 8
    reg_scan = 3 
    read_scan = 8
    total_scan = pcm_scan + aram_scan + x_scan + icache_scan + reg_scan + read_scan
    scan_runtime = 5830 * total_scan
    
    buffer_runtime = clk_period * 100

    runtime = scan_runtime + mac_runtime + buffer_runtime
    return runtime
    
####

cnn0_weights = np.load('./cifar10_conv.npy', allow_pickle=True).item()

#TODO: we zero bias here because we dont use it in tflow.
conv1 = (cnn0_weights['conv1'], cnn0_weights['conv1_bias'].astype(int), cnn0_weights['conv1_scale'])
conv2 = (cnn0_weights['conv2'], cnn0_weights['conv2_bias'].astype(int), cnn0_weights['conv2_scale'])
conv3 = (cnn0_weights['conv3'], cnn0_weights['conv3_bias'].astype(int), cnn0_weights['conv3_scale'])
conv4 = (cnn0_weights['conv4'], cnn0_weights['conv4_bias'].astype(int), cnn0_weights['conv4_scale'])
conv5 = (cnn0_weights['conv5'], cnn0_weights['conv5_bias'].astype(int), cnn0_weights['conv5_scale'])
conv6 = (cnn0_weights['conv6'], cnn0_weights['conv6_bias'].astype(int), cnn0_weights['conv6_scale'])
conv7 = (cnn0_weights['conv7'], cnn0_weights['conv7_bias'].astype(int), cnn0_weights['conv7_scale'])

dense8 = cnn0_weights['dense8']
zeros = np.zeros(shape=(512, 256 - 10))
dense8 = np.concatenate((dense8, zeros), axis=1)

dense8_bias = cnn0_weights['dense8_bias']
zeros = np.zeros(shape=(256 - 10))
dense8_bias = np.concatenate((dense8_bias, zeros))

# we pass scale as a single value, then concatenate inside the layer.
dense8_scale = cnn0_weights['dense8_scale']

dense = (dense8, dense8_bias, dense8_scale)

####
'''
cnn0 = [
Conv(input_size=(32,32,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2, weights=conv1),

Conv(input_size=(32,32,32),  filter_size=(4,4,32,128), stride=2, pad1=1, pad2=1, weights=conv2),
Conv(input_size=(16,16,128), filter_size=(1,1,128,32), stride=1, pad1=0, pad2=0, weights=conv3),

Conv(input_size=(16,16,32), filter_size=(4,4,32,128), stride=2, pad1=1, pad2=1, weights=conv4),
Conv(input_size=(8,8,128),  filter_size=(1,1,128,32), stride=1, pad1=0, pad2=0, weights=conv5),

Conv(input_size=(8,8,32),  filter_size=(4,4,32,128), stride=2, pad1=1, pad2=1, weights=conv6),
Conv(input_size=(4,4,128), filter_size=(1,1,128,32), stride=1, pad1=0, pad2=0, weights=conv7),

Dense(size=(512, 256), weights=dense),
]
'''

cnn1 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(2,2,32,32), stride=2, pad1=0, pad2=0),
# Dense(size=(512, 128)),
# Dense(size=(128, 32)),
]

cnn2 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2, weights=conv1),
Conv(input_size=(8,8,32),  filter_size=(4,4,32,128), stride=2, pad1=1, pad2=1, weights=conv2),
Conv(input_size=(4,4,128), filter_size=(1,1,128,32), stride=1, pad1=0, pad2=0, weights=conv3),
]

'''
# Dense
cnn3 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(4,4,32,32), stride=2, pad1=1, pad2=1),
Dense(size=(512, 256)),
Dense(size=(256, 256))
]
'''

# 32x64, 64x32
cnn4 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(4,4,32,64), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,64), filter_size=(4,4,64,32), stride=2, pad1=1, pad2=1),
]

# 32x64, 64x64
cnn5 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(4,4,32,64), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,64), filter_size=(4,4,64,64), stride=2, pad1=1, pad2=1),
]

# 2x2x64x64, stride 2
cnn6 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(4,4,32,64), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,64), filter_size=(2,2,64,64), stride=2, pad1=0, pad2=0),
]

# 2x2x64x128, stride 2
cnn7 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(4,4,32,64), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,64), filter_size=(2,2,64,128), stride=2, pad1=0, pad2=0),
]

# this will break
# because of zero skipping, we finish too fast and writes get broken.
cnn8 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,64), stride=1, pad1=1, pad2=2),
]

cnn9 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(2,2,32,64), stride=2, pad1=0, pad2=0),
]

cnn10 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(2,2,32,64), stride=1, pad1=0, pad2=1),
# Conv(input_size=(8,8,64), filter_size=(4,4,64,64), stride=2, pad1=1, pad2=1),
]

####

tests = {
# 'cnn0': (3, (32, 32), cnn0),
'cnn1': (3, (8, 8), cnn1),
# 'cnn2': (3, (8, 8), cnn2),
# 'cnn3': (3, (8, 8), cnn3),
# 'cnn4': (3, (8, 8), cnn4),
# 'cnn5': (3, (8, 8), cnn5),
# 'cnn6': (3, (8, 8), cnn6),
# 'cnn7': (3, (8, 8), cnn7),
# 'cnn8': (3, (8, 8), cnn8),
# 'cnn9': (3, (8, 8), cnn9),
# 'cnn10': (3, (8, 8), cnn10)
}

####

for key in tests.keys():
    if not os.path.isdir(key): 
        os.mkdir(key)
    
    num_example, input_shape, model = tests[key]

    x = init_x(num_example, input_shape, 0, 127)
    compile_code(model, './%s' % (key))
    compile_pcm(x, model, './%s' % (key))
    compile_emu(x, model, './%s' % (key))
    emu('./%s' % (key))
    assert (False)

####

def run_command(params):
    (runtime, name) = params
    if   args.compiler == 'iverilog':  
        cmd = "vvp -M. -m ../sim_vpi ../sim_vpi.vvp +run_time=%d +name=%s" % (runtime, name)
    elif args.compiler == 'ncverilog': 
        cmd = 'cd ..; ncverilog -R +access+r +run_time=%d +name=%s -loadvpi ./sim_vpi.so:register_all' % (runtime, './test/' + name)
    else:
        assert (False)
    os.system(cmd)
    return

####

num_runs = len(tests.keys())
parallel_runs = 8

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        name = list(tests.keys())[parallel_run]
        (num_example, input_shape, model) = tests[name]
        runtime = calc_runtime(20, input_shape, model)
        thread_args = (runtime, name)

        t = threading.Thread(target=run_command, args=(thread_args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

####







