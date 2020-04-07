
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

from BB import branch_and_bound

#########################

class Model:
    def __init__(self, layers, params):
        self.layers = layers
        self.params = params
        self.mac_per_array = [2., 2., 2., 2., 2., 2.] # compute this from params
        
        self.nlayer = len(self.layers)
        self.set_dup()

        self.nblock = 0
        for layer in range(self.nlayer):
            nwl, _, nbl, _ = np.shape(self.layers[layer].wb) 
            self.nblock += nwl
                  
    def profile(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        results = {}

        mac_per_array = np.zeros(shape=(num_examples, num_layers))
        mac_per_array_block = np.zeros(shape=(num_examples, self.nblock))
        for example in range(num_examples):
            pred[example] = x[example]
            block1 = 0
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                mac_per_array[example][layer] = result['nmac'] / result['cycle'] / result['array']
                block2 = block1 + self.layers[layer].nwl
                mac_per_array_block[example][block1:block2] = result['nmac'] / result['block_cycle'] / self.layers[layer].factor
                block1 = block2
                
        self.mac_per_array = np.mean(mac_per_array, axis=0)
        self.mac_per_array_block = np.mean(mac_per_array_block, axis=0)
        self.set_dup()

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
        
    def set_dup(self):
        nmac = np.zeros(shape=self.nlayer, dtype=np.int32)
        factor = np.zeros(shape=self.nlayer, dtype=np.int32)
        for layer in range(self.nlayer):
            nmac[layer] = self.layers[layer].nmac
            factor[layer] = self.layers[layer].factor
    
        alloc = branch_and_bound(4096, nmac, factor, self.mac_per_array, self.params)
        assert (np.sum(alloc) <= 4096)
        for layer in range(len(self.layers)):
            dup = alloc[layer] // self.layers[layer].factor
            self.layers[layer].set_dup(dup)

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
            self.w, self.b, self.q = weights['f'], weights['b'], weights['q']
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
        nwl, _, nbl, _ = np.shape(self.wb) 
        self.factor = nwl * nbl
        
        self.nwl = nwl
        self.nbl = nbl
        
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
        
        results['block_cycle'] = metrics[13:]

        nwl, _, nbl, _ = np.shape(self.wb) 
        results['array'] = self.dup * nwl * nbl
        
        print ('narray: %d array: %d nmac %d cycle: %d stall: %d' % (nwl * nbl, results['array'], results['nmac'], results['cycle'], results['stall']))

        return y, results
        
        #########################
        
    def set_dup(self, dup):
        self.dup = dup
        
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
        
        y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], self.dup, self.params)
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


















        
        
        
