
import math
import numpy as np
from conv_utils import conv_output_length
from dot import *
from defines import *

from scipy.stats import norm, binom

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
        self.h, self.w, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.s = stride
        self.p1 = pad1
        self.p2 = pad2
        
        self.y_h = (self.h - self.fh + self.s + self.p1 + self.p2) / self.s
        self.y_w = (self.w - self.fw + self.s + self.p1 + self.p2) / self.s
                
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
        
        # do something like this so we dont need to pass layer around.
        self.params = params.copy()
        self.params['rpr'] = self.rpr()
        
        #########################

    def forward(self, x):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref   = conv_ref(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2)
        y, psum = conv(x=x, f=self.wb, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        # assert (np.all(y == y_ref))
        print (np.min(y - y_ref), np.max(y - y_ref), np.mean(y - y_ref), np.std(y - y_ref))
        return y_ref, psum
        
    '''
    we are thinking this will be a function of:
    1) variance
    2) adc
    3) xbit
    4) weight stats
    5) quantization value
    6) columns (we only do 256 columns at a time)
    7) wbit (if we divide up wbit to different arrays)
    
    break expected error into:
    1) error from variance
    2) error from rpr > adc
    '''
    
    def rpr(self):
        ret = [self.params['adc']] * self.params['bpa']
        if not self.params['skip']:
            return ret
        
        # counting cards:
        # ===============

        # TODO: need to use rpr and adc.
        def calc_e_var(adc, var):
            on = np.array(range(adc + 1)).reshape(1, -1)
            std = np.sqrt(on * var ** 2)

            x = np.array(range(adc + 1)).reshape(-1, 1)
            on_a = np.clip(x - 0.5, 0, adc)
            on_b = np.clip(x + 0.5, 0, adc)
            
            p = norm.cdf(x=on_b, loc=on, scale=std) - norm.cdf(x=on_a, loc=on, scale=std)
            p = np.where(np.isnan(p), 0., p)
            
            e = p * (x - on) ** 2
            e = np.sum(e, axis=0)
            return e
        
        # how many rows are we going to do
        nrow = self.fh * self.fw * self.fc
        
        # weight stats
        wb_cols = np.reshape(self.wb, (self.fh * self.fw * self.fc, self.fn, self.params['bpw']))
        col_density = np.mean(wb_cols, axis=0)
        col_shift = 2 ** np.array(range(self.params['bpw']))

        for rpr in range(self.params['adc'], self.params['adc'] + 4):
            on = np.array(range(0, rpr + 1))
            p = binom.pmf(on, rpr, col_density.reshape(self.fn, self.params['bpw'], 1))
            e_var = calc_e_var(rpr, self.params['sigma'])
            e_rpr = np.where(on > self.params['adc'], on - self.params['adc'], 0)
            e = p * (e_var + e_rpr)
            # e = np.sum(e, axis=2)
            print (np.mean(e, axis=2), np.std(e, axis=2))
            
        assert (False)
            
        for b in range(self.params['bpa']):
            ret[b] = self.params['adc']
        
        return ret 
        
#########################

class Dense(Layer):
    def __init__(self, isize, osize, params, weights=None):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1

        self.isize = isize
        self.osize = osize
        assert((self.osize == 32) or (self.osize == 64) or (self.osize == 128))

        if weights == None:
            maxval = pow(2, params['bpw'] - 1)
            minval = -1 * (maxval - 1)
            values = np.array(range(minval, maxval))
            self.w = np.random.choice(a=values, size=(self.isize, self.osize), replace=True).astype(int)
            self.b = np.zeros(shape=self.osize).astype(int) 
            self.q = 200
        else:
            self.w, self.b, self.q = weights
            # check shape
            assert(np.shape(self.w) == (self.isize, self.osize))
            assert(np.shape(self.b) == (self.osize,))
            assert(np.shape(self.q) == ())
            # cast as int
            self.w = self.w.astype(int)
            self.b = self.b.astype(int)
            self.q = int(self.q)
            # q must be larger than 0
            assert(self.q > 0)
            
        self.params = params.copy()
        self.params['rpr'] = self.rpr()

    def forward(self, x):
        x = np.reshape(x, self.isize)
        y_ref   = dot_ref(x=x, f=self.w, b=self.b, q=self.q)
        y, psum = dot(x=x, f=self.wb, b=self.b, q=self.q, params=self.params)
        assert (np.all(y == y_ref))
        return y_ref, psum

    def rpr(self):
        return [self.params['adc']] * self.params['bpa']

#########################
        
        
        
        
        
        
        
        
        
