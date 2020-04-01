
import math
import numpy as np
import matplotlib.pyplot as plt

from conv_utils import conv_output_length
from cdot import *
from dot_ref import *
from defines import *
from var import *

from scipy.stats import norm, binom
from rpr import rpr

#########################

class Model:
    def __init__(self, layers, params):
        self.layers = layers
        self.params = params
        self.set_ndup()

    def forward(self, x, y):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        results = {}

        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                if (layer in results.keys()):
                    results[layer].append(result)
                else:
                    results[layer] = [result]

        return pred, results
        
    def set_ndup(self):
        if self.params['skip']:
            self.set_ndup2()
        else:
            self.set_ndup1()
        
    #######################################################
    
    def profile(self, x):
        '''
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        results = {}

        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                if (layer in results.keys()):
                    results[layer].append(result)
                else:
                    results[layer] = [result]

        return pred, results
        '''
        # setup a profile function so we arnt hardcoding some random #s.
        # why do we use MAC/cycle and not x_non_zero, shouldn't they be the same ?
        pass
    
    #######################################################

    def set_ndup1(self):
        nmac = 0
        for layer in self.layers:
            nmac += layer.nmac

        for layer in range(len(self.layers)):
            p = self.layers[layer].nmac / nmac
            if (layer == 0): ndup = p * (1024 * 128 * 128) / np.prod(np.shape(self.layers[layer].wb)) * (128 / 27)
            else:            ndup = p * (1024 * 128 * 128) / np.prod(np.shape(self.layers[layer].wb))
            self.layers[layer].set_ndup(int(np.ceil(ndup)))

    def set_ndup2(self):
    
        x_non_zero = np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])
            
        ###########
    
        shares = np.zeros(shape=len(self.layers))
        for layer in range(len(self.layers)):
            fh, fw, fc, fn = np.shape(self.layers[layer].w)
            # rows_per_array = min(fh * fw * fc, 128)
            rows_per_array = 128 
            cycle_per_array = rows_per_array * x_non_zero[layer]
            shares[layer] = self.layers[layer].nmac * cycle_per_array
    
        ###########

        total_weights = 1024 * 128 * 128
        
        nmac = 0
        for layer in self.layers:
            nmac += layer.nmac
            
        ###########

        shares = shares / np.sum(shares)
        
        for layer in range(len(self.layers)):
            # layer_weights = np.prod(np.shape(self.layers[layer].w)) * 8
            layer_weights = np.prod(np.shape(self.layers[layer].wb))
            share = shares[layer] * total_weights / layer_weights
            ndup = int(np.round(share))
            self.layers[layer].set_ndup(ndup)
            # print (ndup)

    def set_ndup3(self):
    
        mac_per_array = np.array([3.5, 6.3, 7.8, 9.6, 9.4, 16.])
            
        ###########
    
        shares = np.zeros(shape=len(self.layers))
        for layer in range(len(self.layers)):
            fh, fw, fc, fn = np.shape(self.layers[layer].w)
            # rows_per_array = min(fh * fw * fc, 128)
            rows_per_array = 128 
            cycle_per_array = rows_per_array / mac_per_array[layer]
            shares[layer] = self.layers[layer].nmac * cycle_per_array

        ###########

        total_weights = 1024 * 128 * 128
        
        nmac = 0
        for layer in self.layers:
            nmac += layer.nmac
            
        ###########
        
        shares = shares / np.sum(shares)
        
        for layer in range(len(self.layers)):
            # layer_weights = np.prod(np.shape(self.layers[layer].w)) * 8
            layer_weights = np.prod(np.shape(self.layers[layer].wb))
            share = shares[layer] * total_weights / layer_weights
            ndup = int(np.round(share))
            self.layers[layer].set_ndup(ndup)
            # print (ndup)

    #######################################################            

#########################

class Layer:
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)

    def rpr(self):
        assert(False)
        
#########################

class Conv(Layer):
    def __init__(self, input_size, filter_size, pool, stride, pad1, pad2, params, weights=None):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1

        self.input_size = input_size
        self.xh, self.xw, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.p = pool
        self.s = stride
        self.p1 = pad1
        self.p2 = pad2

        assert (self.s == 1)
        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw)

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

        self.params = params.copy()
        self.params['var'] = lut_var(params['sigma'], 32)
        
        #########################
        
        w_offset = self.w + params['offset']
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
        wb = np.stack(wb, axis=-1)

        wb_cols = np.reshape(wb, (self.fh * self.fw * self.fc, self.fn, params['bpw']))
        col_density = np.mean(wb_cols, axis=0)

        nrow = self.fh * self.fw * self.fc
        p = np.max(col_density, axis=0)
        self.params['rpr'] = rpr(nrow=nrow, p=p, q=self.q, params=self.params)
        
        #########################
        
        self.wb = self.cut()
        
        #########################

    def forward(self, x):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2)
        # y, metrics = cconv(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        y, metrics = self.conv(x=x)

        y_min = np.min(y - y_ref)
        y_max = np.max(y - y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        assert (self.s == 1)
        nmac = (self.xh * self.xw) * (self.fh * self.fw * self.fc * self.fn)
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        results = {}
        results['nmac']  = nmac
        results['adc']   = metrics[0:8]
        results['cycle'] = metrics[8]
        results['ron']   = metrics[9]
        results['roff']  = metrics[10]
        results['wl']    = metrics[11]
        results['std']   = y_std
        results['mean']  = y_mean
        results['stall'] = metrics[12]

        nwl, _, nbl, _ = np.shape(self.wb) 
        results['array'] = self.ndup * nwl * nbl
        # print (results['array'])
        
        print ('array: %d nmac %d cycle: %d stall: %d' % (results['array'], results['nmac'], results['cycle'], results['stall']))

        return y, results
        
        #########################
        
    def set_ndup(self, ndup):
        self.ndup = ndup
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        #########################
        
        x = np.pad(array=x, pad_width=[[self.p1,self.p2], [self.p1,self.p2], [0,0]], mode='constant')
        patches = []
        for h in range(yh):
            for w in range(yw):
                patch = np.reshape(x[h*self.s:(h*self.s+self.fh), w*self.s:(w*self.s+self.fw), :], -1)
                patches.append(patch)
                
        #########################
        
        patches = np.stack(patches, axis=0)
        pb = []
        for xb in range(self.params['bpa']):
            pb.append(np.bitwise_and(np.right_shift(patches.astype(int), xb), 1))
        
        patches = np.stack(pb, axis=-1)
        npatch, nrow, nbit = np.shape(patches)
        
        #########################
        
        # print (np.count_nonzero(patches) / np.prod(np.shape(patches)))
        # print (8 * np.count_nonzero(patches, axis=(0, 1)) / np.prod(np.shape(patches)))
        
        #########################
        
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(npatch, self.params['wl'] - (nrow % self.params['wl']), self.params['bpa']))
            patches = np.concatenate((patches, zeros), axis=1)
            
        patches = np.reshape(patches, (npatch, -1, self.params['wl'], self.params['bpa']))
        
        #########################
        
        y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], self.ndup, self.params)
        y = np.reshape(y, (yh, yw, self.fn))
        
        assert(np.all(np.absolute(y) < 2 ** 23))
        y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.floor(y)
        y = np.clip(y, -128, 127)

        return y, metrics
        
    def cut(self):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        w_offset = self.w + self.params['offset']
        w_matrix = np.reshape(w_offset, (self.fh * self.fw * self.fc, self.fn))
        wb = []
        for bit in range(self.params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_matrix, bit), 1))
        wb = np.stack(wb, axis=-1)
        
        ########################
        
        nrow, ncol, nbit = np.shape(wb)
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(self.params['wl'] - (nrow % self.params['wl']), ncol, nbit))
            wb = np.concatenate((wb, zeros), axis=0)

        nrow, ncol, nbit = np.shape(wb)
        wb = np.reshape(wb, (-1, self.params['wl'], ncol, nbit))
        
        ########################

        nwl, wl, ncol, nbit = np.shape(wb)
        wb = np.transpose(wb, (0, 1, 3, 2))
        wb = np.reshape(wb, (nwl, self.params['wl'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['wl'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['wl'], -1, self.params['bl']))

        ########################

        return wb
        
#########################


















        
        
        
