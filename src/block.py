
import math
import numpy as np
import matplotlib.pyplot as plt

from var import *

from layers import *
from conv import *
   
#########################

class Block1(Layer):
    def __init__(self, input_size, filter_size, stride, params, weights):
        self.input_size = input_size
        self.f1, self.f2 = filter_size

        input_size1 = input_size
        input_size2 = (input_size1[0] // stride, input_size1[1] // stride, input_size1[2])

        self.conv1 = Conv(input_size1, (3,3,self.f1,self.f2), 1, stride, pad1=1, pad2=1, params=params, weights=weights)
        self.conv2 = Conv(input_size2, (3,3,self.f2,self.f2), 1, 1,      pad1=1, pad2=1, params=params, weights=weights, relu_flag=False)

        self.layer_id = Layer.layer_id
        Layer.layer_id += 1

        self.s = weights[self.layer_id]['s']
        self.s2 = weights[self.layer_id]['s2']
        self.q = weights[self.layer_id]['q']

    def init(self, params):
        self.params = params.copy()
        self.conv1.init(self.params)
        self.conv2.init(self.params)
        
    def set_profile_adc(self, counts):
        self.conv1.set_profile_adc(counts)
        self.conv2.set_profile_adc(counts)

    def profile_adc(self, x):
        y1, r1 = self.conv1.profile_adc(x)
        y2, r2 = self.conv2.profile_adc(y1)
        y3 = relu(x + y2)
        
        result = {}
        result.update(r1)
        result.update(r2)
        return y3, result 

    def act(self, x):
        out = x
        out = out / self.q
        out = np.round(out)
        out = np.clip(out, -128, 127)
        return out

    def forward(self, x, x_ref, profile=False):
        y1, y1_ref, r1 = self.conv1.forward(x,  x_ref,  profile=profile)
        y2, y2_ref, r2 = self.conv2.forward(y1, y1_ref, profile=profile)

        y3     = self.act(relu(self.s * x     + self.s2 * y2))
        y3_ref = self.act(relu(self.s * x_ref + self.s2 * y2_ref))

        result = []
        result.extend(r1)
        result.extend(r2)
        return y3, y3_ref, result 

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return [self.conv1, self.conv2]
        
#############

class Block2(Layer):
    def __init__(self, input_size, filter_size, stride, params, weights):
        self.input_size = input_size
        self.f1, self.f2 = filter_size
        
        assert (self.f1 == input_size[2])
        input_size1 = input_size
        input_size2 = (input_size1[0] // stride, input_size1[1] // stride, self.f2)

        self.conv1 = Conv(input_size1, (3,3,self.f1,self.f2), 1, stride, pad1=1, pad2=1, params=params, weights=weights)
        self.conv2 = Conv(input_size2, (3,3,self.f2,self.f2), 1, 1,      pad1=1, pad2=1, params=params, weights=weights, relu_flag=False, quantize_flag=False)
        self.conv3 = Conv(input_size1, (1,1,self.f1,self.f2), 1, stride, pad1=0, pad2=0, params=params, weights=weights, relu_flag=False, quantize_flag=False)

        self.layer_id = Layer.layer_id
        Layer.layer_id += 1

        self.s2 = weights[self.layer_id]['s2']
        self.s3 = weights[self.layer_id]['s3']
        self.q = weights[self.layer_id]['q']

    def init(self, params):
        self.params = params.copy()
        self.conv1.init(self.params)
        self.conv2.init(self.params)
        self.conv3.init(self.params)
        
    def set_profile_adc(self, counts):
        self.conv1.set_profile_adc(counts)
        self.conv2.set_profile_adc(counts)
        self.conv3.set_profile_adc(counts)

    def profile_adc(self, x):
        y1, r1 = self.conv1.profile_adc(x)
        y2, r2 = self.conv2.profile_adc(y1)
        y3, r3 = self.conv3.profile_adc(x)
        y4 = relu(y2 + y3)
        
        result = {}
        result.update(r1)
        result.update(r2)
        result.update(r3)
        return y4, result 

    def act(self, x):
        out = x
        out = out / self.q
        out = np.round(out)
        out = np.clip(out, -128, 127)
        return out

    def forward(self, x, x_ref, profile=False):
        y1, y1_ref, r1 = self.conv1.forward(x,  x_ref,  profile=profile)
        y2, y2_ref, r2 = self.conv2.forward(y1, y1_ref, profile=profile)
        y3, y3_ref, r3 = self.conv3.forward(x,  x_ref,  profile=profile)

        y4     = self.act(relu(self.s2 * y2     + self.s3 * y3))
        y4_ref = self.act(relu(self.s2 * y2_ref + self.s3 * y3_ref))

        result = []
        result.extend(r1)
        result.extend(r2)
        result.extend(r3)
        return y4, y4_ref, result 

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return [self.conv1, self.conv2, self.conv3]
        
#############







        
        
        
