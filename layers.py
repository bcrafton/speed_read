
import math
import numpy as np
from conv_utils import conv_output_length
from dot import *
from defines import *

#########################

class Conv:
    def __init__(self, input_size, filter_size, stride, pad1, pad2, weights=None):
        self.input_size = input_size
        self.h, self.w, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.stride = stride
        self.pad1 = pad1
        self.pad2 = pad2
        
        self.y_h = (self.h - self.fh + self.stride + self.pad1 + self.pad2) / self.stride
        self.y_w = (self.w - self.fw + self.stride + self.pad1 + self.pad2) / self.stride
        
        if (self.fh == 1): 
            assert((self.stride==1) and (self.pad1==0) and (self.pad2==0))

        if weights == None:
            values = np.array(range(-6, 8))
            self.weights = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            # np.random.choice(a=values, size=self.fn, replace=True).astype(int)
            self.bias = np.zeros(shape=self.fn).astype(int)
            # make me a lut function based based on input size
            self.quant = np.ones(shape=self.fn).astype(int) * 200 
        else:
            self.weights, self.bias, self.quant = weights
            assert(np.shape(self.weights) == self.filter_size)
            assert(np.shape(self.bias) == (self.fn,))
            assert(np.shape(self.quant) == ())

            self.quant = np.ones(shape=self.fn) * self.quant

            self.weights = self.weights.astype(int)
            self.bias = self.bias.astype(int)
            self.quant = self.quant.astype(int)

    def forward(self, x, params):
        # could move ref inside conv.
        # conv(x, w_params, op_params, pim_params) rather than all the args 
        y_ref = conv_ref(x=x, f=self.weights, b=self.bias, q=self.quant, stride=self.stride, pad1=self.pad1, pad2=self.pad2)
        y     = conv(x=x, f=self.weights, b=self.bias, q=self.quant, stride=self.stride, pad1=self.pad1, pad2=self.pad2, params=params)
        assert (np.all(y == y_ref))
        return y

#########################

class Dense:
    def __init__(self, size, weights=None):        
        self.size = size
        self.input_size, self.output_size = self.size
        assert((self.output_size == 32) or (self.output_size == 64) or (self.output_size == 128))

        if weights == None:
            values = np.array(range(-1, 4))
            self.weights = np.random.choice(a=values, size=self.size, replace=True).astype(int)
            # np.random.choice(a=values, size=self.output_size, replace=True).astype(int)
            self.bias = np.zeros(shape=self.output_size).astype(int) 
            # make lut function based on input size
            self.quant = np.ones(shape=self.output_size).astype(int) * 200
        else:
            self.weights, self.bias, self.quant = weights
            assert(np.shape(self.weights) == self.size)
            assert(np.shape(self.bias) == (self.output_size,))
            assert(np.shape(self.quant) == ())
            self.quant = np.ones(shape=self.output_size) * self.quant

            self.weights = self.weights.astype(int)
            self.bias = self.bias.astype(int)
            self.quant = self.quant.astype(int)

    def forward(self, x, params):
        x = np.reshape(x, self.input_size)
        # could move ref inside dot.
        # dot(x, w_params, op_params, pim_params) rather than all the args 
        y_ref = dot_ref(x=x, f=self.weights, b=self.bias, q=self.quant)
        y     = dot(x=x, f=self.weights, b=self.bias, q=self.quant, params=params)
        assert (np.all(y == y_ref))
        return y

#########################
        
        
        
        
        
        
        
        
        
