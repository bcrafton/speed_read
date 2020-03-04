
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
from emu import emu
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
    
def compile_emu(x, model, path):
    emu = {}

    nlayers = len(model)
    for l in range(nlayers):
        if (model[l].opcode() == OPCODE_CONV):
            emu[l] = {'weights': model[l].weights, 'bias': model[l].bias, 'quant': model[l].quant,
                          'op': model[l].opcode(), 'x': model[l].input_size, 
                          'dims': {'stride': model[l].stride, 'pad1': model[l].pad1, 'pad2': model[l].pad2}}
        else:
            emu[l] = {'weights': model[l].weights, 'bias': model[l].bias, 'quant': model[l].quant,
                          'op': model[l].opcode(), 'x': model[l].input_size}

    n, h, w, c = np.shape(x)
    emu['x'] = x
    emu['num_example'] = n
    emu['num_layer'] = nlayers
    np.save("%s/emu" % (path), emu)
    
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
    if not os.path.isdir(key): 
        os.mkdir(key)
    
    num_example, input_shape, model = tests[key]

    x = init_x(num_example, input_shape, 0, 127)
    compile_emu(x, model, './%s' % (key))
    emu('./%s' % (key))

####








