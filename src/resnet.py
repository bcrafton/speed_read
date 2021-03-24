
import numpy as np
import tensorflow as tf

from layers import *
from conv import *
from block import *
from model import *
from dense import *

################

def quantize_np(x):
  scale = 127 / np.max(np.absolute(x))
  x = x * scale
  x = np.round(x)
  x = np.clip(x, -127, 127)
  return x, scale

def load_inputs(num_example):
    dataset = np.load('../imagenet.npy', allow_pickle=True).item()
    xs, ys = dataset['x'], dataset['y']
    ys = np.argmax(ys, axis=1)
    assert (np.shape(xs) == (10, 224, 224, 3))

    # TODO: make sure we are using the right input images and weights
    # xs = xs / 255. 
    # xs = xs - np.array([0.485, 0.456, 0.406])
    # xs = xs / np.array([0.229, 0.224, 0.225])
    # xs, scale = quantize_np(xs)

    # print (np.max(xs), np.min(xs))
    # assert (False)
    # xs = xs // 2
    # xs = xs.astype(int)

    xs = xs[0:num_example]
    ys = ys[0:num_example]
    return xs, ys

################

def create_model(array_params):
    weights = np.load('../resnet18_weights1.npy', allow_pickle=True).item()

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

    # AvgPool(input_size=(7, 7, 512), kernel_size=7, stride=7, params=array_params, weights=weights),
    # Dense(size=(512, 1000), params=array_params, weights=weights)
    ]

    model = Model(layers=layers, array_params=array_params)
    return model

################

def load_resnet(num_example, array_params):
    model = create_model(array_params)
    x, y = load_inputs(num_example)
    return model, x, y

################























