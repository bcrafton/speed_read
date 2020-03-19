
import math
import numpy as np
import matplotlib.pyplot as plt

from conv_utils import conv_output_length
from dot import *
from cdot import *
from dot_ref import *
from defines import *

from scipy.stats import norm, binom
from rpr import rpr

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        y = [None] * num_examples
        results = {}

        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(num_layers):
                y[example], result = self.layers[layer].forward(x=y[example])
                if (layer in results.keys()):
                    results[layer].append(result)
                else:
                    results[layer] = [result]

        return y, results

class Layer:
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)

    def rpr(self):
        assert(False)

class Conv(Layer):
    def __init__(self, input_size, filter_size, stride, pad1, pad2, params, weights=None):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1

        self.input_size = input_size
        self.xh, self.xw, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.s = stride
        self.p1 = pad1
        self.p2 = pad2
        
        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        self.yw = (self.xw - self.fw + self.s + self.p1 + self.p2) // self.s
                
        if (self.fh == 1): 
            assert((self.s==1) and (self.p1==0) and (self.p2==0))

        maxval = pow(2, params['bpw'] - 1)
        minval = -1 * maxval
        if weights == None:
            values = np.array(range(minval + 1, maxval))
            self.w = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            self.b = np.zeros(shape=self.fn).astype(int)
            self.q = 200
        else:
            self.w, self.b, self.q = weights
            assert (np.all(self.w >= minval))
            assert (np.all(self.w <= maxval))
            # check shape
            assert(np.shape(self.w) == self.filter_size)
            assert(np.shape(self.b) == (self.fn,))
            assert(np.shape(self.q) == ())
            # cast as int
            self.w = self.w.astype(int)
            self.b = self.b.astype(int)
            self.q = int(self.q)
            # q must be larger than 0
            assert(self.q > 0)

        #########################

        # w_offset = self.w + pow(2, params['bpw'] - 1)
        w_offset = self.w + params['offset']
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
        self.wb = np.stack(wb, axis=-1)
        
        #########################
        
        self.params = params.copy()

        w_offset = self.w + params['offset']
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
        wb = np.stack(wb, axis=-1)

        wb_cols = np.reshape(self.wb, (self.fh * self.fw * self.fc, self.fn, params['bpw']))
        col_density = np.mean(wb_cols, axis=0)

        nrow = self.fh * self.fw * self.fc
        p = np.max(col_density, axis=0)
        self.params['rpr'] = rpr(nrow=nrow, p=p, q=self.q, params=self.params)
        
        #########################

    def forward(self, x):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref   = conv_ref(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2)
        # y, psum = conv(x=x, f=self.wb, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        y, psum = cconv(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)

        y_min = np.min(y - y_ref)
        y_max = np.max(y - y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)

        nmac = (self.yh * self.yw) * (self.fh * self.fw * self.fc * self.fn)

        '''
        print (np.mean(y_ref), np.std(y_ref))
        print (np.mean(self.w), np.std(self.w))
        print (self.q)
        print ()
        '''

        # print (np.mean(y), np.std(y))
        # print (self.params['cards'], (self.yh, self.yh), (self.fh, self.fw, self.fc, self.fn), 'mac / cycle', nmac / psum, np.mean(y_ref), np.std(y_ref))

        # plt.hist(y_ref.flatten(), bins=128)
        # plt.show()

        return y_ref, [nmac / psum, y_mean, y_std]
        
#########################
        
        
        
        
        
        
        
        
        
