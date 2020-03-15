
import math
import numpy as np
import matplotlib.pyplot as plt

from conv_utils import conv_output_length
from dot import *
from cdot import *
from dot_ref import *
from defines import *

from scipy.stats import norm, binom

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        y = [None] * num_examples
        metrics = {}

        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(num_layers):
                y[example], metric = self.layers[layer].forward(x=y[example])
                metrics[(example, layer)] = metric

        return y, metrics

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
        # y, psum = conv(x=x, f=self.wb, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        y, psum = cconv(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)

        y_min = np.min(y - y_ref)
        y_max = np.max(y - y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)

        return y_ref, {'psum': psum, 'y_min': y_min, 'y_max': y_max, 'y_mean': y_mean, 'y_std': y_std}
        
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
    # TODO: move me into my own file, dont make me belong to Conv class.
    def rpr(self):
        rpr_lut = {}
    
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                rpr_lut[(xb, wb)] = self.params['adc']
            
        if not (self.params['skip'] and self.params['cards']):
            for key in sorted(rpr_lut.keys()):
                print (key, rpr_lut[key])
                
            # print (np.average(list(rpr_lut.values())))

            return rpr_lut
        
        # counting cards:
        # ===============
        
        def prob_err(p, var, adc, rpr, row):
            def prob_err_help(e, p, var, adc, rpr):
                psum = 0
                for s in range(1, rpr + 1):
                    bin = binom.pmf(s, rpr, p)
                    psum += ((s + e) < adc) * bin * (norm.cdf(e + 0.5, 0, var * np.sqrt(s)) - norm.cdf(e - 0.5, 0, var * np.sqrt(s)))
                    psum += ((s + e) == adc) * bin * (1 - norm.cdf(adc - s - 0.5, 0, var * np.sqrt(s)))

                # zero case:
                psum += ((e - 0.5 < 0) * (0 < e + 0.5)) * binom.pmf(0, rpr, p)
                return psum
            
            s = np.array(range(-rpr, rpr+1))
            pe = prob_err_help(s, p, var, adc, rpr)
            mu = np.sum(pe * s)
            std = np.sqrt(np.sum(pe * (s - mu) ** 2))
            
            mu = mu * row
            std = np.sqrt(std ** 2 * row)
            return mu, std

        # how many rows are we going to do
        nrow = self.fh * self.fw * self.fc
        
        # weight stats
        wb_cols = np.reshape(self.wb, (self.fh * self.fw * self.fc, self.fn, self.params['bpw']))
        col_density = np.mean(wb_cols, axis=0)

        rpr_lut = {}
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                # rpr_low = max(1, self.params['adc'] // 2)
                # rpr_high = 2 * self.params['adc']
                rpr_low = 2
                rpr_high = 16
                for rpr in range(rpr_low, rpr_high + 1):
                    scale = 2**(wb - 1) * 2**(xb - 1)
                    p = np.max(col_density[:, wb])
                    mu, std = prob_err(p, self.params['sigma'], self.params['adc'], rpr, np.ceil(nrow / rpr))
                    e = (scale / self.q) * 5 * std
                    # print (wb, xb, rpr, std, e)
                    
                    if rpr == rpr_low:
                        rpr_lut[(xb, wb)] = rpr
                    if e < 1.:
                        rpr_lut[(xb, wb)] = rpr

                    # print ('(%d %d %d %d) : %f %f %f' % (self.layer_id, wb, xb, rpr, scale, self.q, var))
  
        '''
        for key in sorted(rpr_lut.keys()):
            print (key, rpr_lut[key])
        '''
        
        return rpr_lut

#########################
        
        
        
        
        
        
        
        
        
