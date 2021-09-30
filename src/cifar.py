
import numpy as np
import tensorflow as tf

from layers import *
from conv import *
from block import *
from model import *
from dense import *

################

def load_inputs(num_example, input_shape, xlow, xhigh):
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

################

def create_model(array_params):
    weights = np.load('../cifar10_weights.npy', allow_pickle=True).item()

    layers = [
    Conv(input_size=(32,32, 3), filter_size=(3,3, 3,64), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #Conv(input_size=(32,32,64), filter_size=(3,3,64,64), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #AvgPool(input_size=(32, 32, 64), kernel_size=2, stride=2, params=array_params, weights=weights),

    #Conv(input_size=(16,16,64),  filter_size=(3,3, 64,128), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #Conv(input_size=(16,16,128), filter_size=(3,3,128,128), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #AvgPool(input_size=(16, 16, 128), kernel_size=2, stride=2, params=array_params, weights=weights),

    #Conv(input_size=(8,8,128), filter_size=(3,3,128,256), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #Conv(input_size=(8,8,256), filter_size=(3,3,256,256), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #AvgPool(input_size=(8, 8, 128), kernel_size=2, stride=2, params=array_params, weights=weights),

    #Conv(input_size=(4,4,256), filter_size=(3,3,256,256), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),
    #Conv(input_size=(4,4,256), filter_size=(3,3,256,256), pool=1, stride=1, pad1=1, pad2=1, params=array_params, weights=weights),

    #AvgPool(input_size=(4, 4, 256), kernel_size=4, stride=4, params=array_params, weights=weights),
    #Dense(size=(256, 10), params=array_params, weights=weights)
    ]

    model = Model(layers=layers, array_params=array_params)
    return model

################

def load_cifar(num_example, array_params):
    model = create_model(array_params)
    x, y = load_inputs(num_example, (32, 32), 0, 127)
    return model, x, y

################























