
import math
import numpy as np
import matplotlib.pyplot as plt

from defines import *
from var import *

from layers import *
from conv import *
   
#########################

class Block1(Layer):
    def __init__(self, input_size, filter_size, stride, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.f1, self.f2 = filter_size
        self.s = stride
        
        self.params = params.copy()

        input_size1 = input_size
        input_size2 = (input_size1[0] // stride, input_size1[1] // stride, input_size1[2])

        # Conv(input_size=(224, 224, 3), filter_size=(7,7,3,64), pool=1, stride=2, pad1=3, pad2=3, params=params, weights=weights[0]),
        self.conv1 = Conv(input_size1, (3,3,self.f1,self.f2), 1, stride, pad1=1, pad2=1, params=params, weights=weights)
        self.conv2 = Conv(input_size2, (3,3,self.f2,self.f2), 1, 1,      pad1=1, pad2=1, params=params, weights=weights, relu_flag=False)

    def forward(self, x):
        y1, r1 = self.conv1.forward(x)
        y2, r2 = self.conv2.forward(y1)        
        y3 = relu(x + y2)
        
        result = []
        result.extend(r1)
        result.extend(r2)
        return y3, result 

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return [self.conv1, self.conv2]
        
#############

class Block2(Layer):
    def __init__(self, input_size, filter_size, stride, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.f1, self.f2 = filter_size
        self.s = stride
        
        self.params = params.copy()

        assert (self.f1 == input_size[2])
        input_size1 = input_size
        input_size2 = (input_size1[0] // stride, input_size1[1] // stride, self.f2)

        # Conv(input_size=(224, 224, 3), filter_size=(7,7,3,64), pool=1, stride=2, pad1=3, pad2=3, params=params, weights=weights[0]),
        self.conv1 = Conv(input_size1, (3,3,self.f1,self.f2), 1, stride, pad1=1, pad2=1, params=params, weights=weights)
        self.conv2 = Conv(input_size2, (3,3,self.f2,self.f2), 1, 1,      pad1=1, pad2=1, params=params, weights=weights, relu_flag=False)
        self.conv3 = Conv(input_size1, (1,1,self.f1,self.f2), 1, stride, pad1=0, pad2=0, params=params, weights=weights, relu_flag=False)

    def forward(self, x):
        y1, r1 = self.conv1.forward(x)
        y2, r2 = self.conv2.forward(y1)
        y3, r3 = self.conv3.forward(x)
        y4 = relu(y2 + y3)
        
        result = []
        result.extend(r1)
        result.extend(r2)
        result.extend(r3)
        return y4, result 

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return [self.conv1, self.conv2, self.conv3]
        
#############







        
        
        
