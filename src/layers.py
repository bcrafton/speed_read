
import math
import numpy as np
import matplotlib.pyplot as plt

from var import *
from conv_utils import *

from scipy.stats import norm, binom

from AA import array_allocation

#########################

class Layer:
    layer_id = 0
    weight_id = 0
    
    def __init__(self):
        assert(False)
        
    def forward(self, x, x_ref, profile=False):
        assert(False)
        
    def init(self, params):
        pass
        
    def set_profile_adc(self, counts):
        pass
        
    def profile_adc(self, x):
        y, _, _ = self.forward(x=x, x_ref=x)
        return y, {}
        
    def nblock(self):
        return 0

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return []
        
#########################

class AvgPool(Layer):
    def __init__(self, input_size, kernel_size, stride, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.k = kernel_size
        self.s = stride
        
        assert (self.k == self.s)

    # max pool and avg pool mess things up a bit because they dont do the right padding in tensorflow.
    def forward(self, x, x_ref, profile=False):
        # cim
        y = avg_pool(x, self.k)
        y = np.clip(np.floor(y), -128, 127)
        # ref
        y_ref = avg_pool(x_ref, self.k)
        y_ref = np.clip(np.floor(y_ref), -128, 127)
        # return
        assert (np.all((y % 1) == 0))
        assert (np.all((y_ref % 1) == 0))
        return y, y_ref, []

#############

class MaxPool(Layer):
    def __init__(self, input_size, kernel_size, stride, params, weights):
        assert False, "Integrate + Verify conv_utils/max_pool()"

        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.xh, self.xw, self.xc = self.input_size
        
        self.k = kernel_size
        self.pad = self.k // 2
        self.s = stride
        
        self.output_size = input_size[0] // stride, input_size[1] // stride, input_size[2]
        self.yh, self.yw, self.yc = self.output_size

    def forward(self, x, profile=False):
        # max pool and avg pool mess things up a bit because they dont do the right padding in tensorflow.
        x = np.pad(array=x, pad_width=[[self.pad,self.pad], [self.pad,self.pad], [0,0]], mode='constant')
        y = np.zeros(shape=self.output_size)
        for h in range(self.yh):
            for w in range(self.yw):
                y[h, w, :] = np.max( x[h*self.s:(h*self.s+self.k), w*self.s:(w*self.s+self.k), :], axis=(0, 1) )

        return y, []

#############






        
        
        
