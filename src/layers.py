
import math
import numpy as np
from conv_utils import conv_output_length
from dot import *
from defines import *

#########################

class Conv:
    def __init__(self, input_size, filter_size, stride, pad1, pad2, params, weights=None):
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
        
        self.params = params
        
        if (self.fh == 1): 
            assert((self.s==1) and (self.p1==0) and (self.p2==0))

        if weights == None:
            maxval = pow(2, params['bpw'] - 1)
            minval = -1 * (maxval - 1)
            values = np.array(range(minval, maxval))
            self.w = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            self.b = np.zeros(shape=self.fn).astype(int)
            self.q = 200 
        else:
            self.w, self.b, self.q = weights
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

        w_offset = self.w + pow(2, params['bpw'] - 1)
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
        self.wb = np.stack(wb, axis=-1)
        
        nonzero = np.count_nonzero(self.wb) / np.prod(np.shape(self.wb))
        print ('nonzero %', nonzero)

    def forward(self, x):
        # could move ref inside conv.
        y_ref   = conv_ref(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2)
        y, psum = conv(x=x, f=self.wb, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        assert (np.all(y == y_ref))
        return y, psum

#########################

class Dense:
    def __init__(self, isize, osize, params, weights=None):
        self.isize = isize
        self.osize = osize
        assert((self.osize == 32) or (self.osize == 64) or (self.osize == 128))

        self.params = params

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

    def forward(self, x):
        x = np.reshape(x, self.isize)
        # could move ref inside dot.
        y_ref   = dot_ref(x=x, f=self.w, b=self.b, q=self.q)
        y, psum = dot(x=x, f=self.wb, b=self.b, q=self.q, params=self.params)
        assert (np.all(y == y_ref))
        return y, psum

#########################
        
        
        
        
        
        
        
        
        
