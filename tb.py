
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
from sim import sim
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
    
def save_params(x, model, path):
    params = {}

    nlayers = len(model)
    n, h, w, c = np.shape(x)
    
    params['x'] = x
    params['num_example'] = n
    params['num_layer'] = nlayers
    
    for l in range(nlayers):
        if (model[l].opcode() == OPCODE_CONV):
            params[l] = { 'weights': model[l].weights, 
                          'bias': model[l].bias, 
                          'quant': model[l].quant,
                          'op': model[l].opcode(), 
                          'input_size': model[l].input_size, 
                          'dims': {'stride': model[l].stride, 'pad1': model[l].pad1, 'pad2': model[l].pad2} }
        else:
            params[l] = { 'weights': model[l].weights, 
                          'bias': model[l].bias, 
                          'quant': model[l].quant,
                          'op': model[l].opcode(), 
                          'input_size': model[l].input_size }

    np.save("%s/params" % (path), params)
    
####

cnn1 = [
Conv(input_size=(8,8,3), filter_size=(4,4,3,32), stride=1, pad1=1, pad2=2),
Conv(input_size=(8,8,32), filter_size=(2,2,32,32), stride=2, pad1=0, pad2=0),
]

####

tests = {
'cnn1': (3, (8, 8), cnn1)
}

####

for key in tests.keys():
    path = './sims/%s' % (key)
    if not os.path.isdir(path): 
        os.mkdir(path)
    
    num_example, input_shape, model = tests[key]

    x = init_x(num_example, input_shape, 0, 127)
    save_params(x, model, path)
    sim(path)

####








