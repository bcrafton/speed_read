
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
from AA import array_allocation

#########################

class Model:
    def __init__(self, layers, params):
        self.layers = layers
        self.params = params

        self.nlayer = len(self.layers)

        self.weights = []
        for layer in self.layers:
            self.weights.extend(layer.weights())
        self.nweight = len(self.weights)

        self.block_map = []
        self.nblock = 0
        for w, weight in enumerate(self.weights):
            self.block_map.append(slice(self.nblock, self.nblock + weight.nwl))
            self.nblock += weight.nwl

        self.mac_per_array_layer = [2.] * self.nweight
        self.set_layer_alloc()
        
        print ('nblock', self.nblock)
        self.mac_per_array_block = [2.] * self.nblock
        self.set_block_alloc()

    def profile(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        results = {}

        mac_per_array_layer = np.zeros(shape=(num_examples, self.nweight))
        mac_per_array_block = np.zeros(shape=(num_examples, self.nblock))
        
        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                assert (np.all((pred[example] % 1) == 0))
                for r in result:
                    mac_per_array_layer[example][r['id']] = (r['nmac'] / self.weights[r['id']].factor) / (r['cycle'] * self.weights[r['id']].layer_alloc)
                    mac_per_array_block[example][self.block_map[r['id']]] = (r['nmac'] / self.weights[r['id']].factor) / (r['block_cycle'])
                    
        self.mac_per_array_layer = np.mean(mac_per_array_layer, axis=0)
        self.mac_per_array_block = np.mean(mac_per_array_block, axis=0)
        
        if self.params['alloc'] == 'layer': 
            self.set_layer_alloc() # block alloc was failing when layer was selected, this is a bandaid.
        else:
            self.set_block_alloc()

    def forward(self, x, y):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        # results = [[] for weight in range(self.nweight)] 
        results = {}
        for weight in range(self.nweight):
            results[weight] = []

        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                assert (np.all((pred[example] % 1) == 0))
                for r in result:
                    results[r['id']].append(r)
                    
        results['block_mac'] = self.mac_per_array_block
        results['layer_mac'] = self.mac_per_array_layer

        return pred, results

    def set_layer_alloc(self):
        nmac = np.zeros(shape=self.nweight, dtype=np.int32)
        factor = np.zeros(shape=self.nweight, dtype=np.int32)
        for weight in range(self.nweight):
            nmac[weight] = self.weights[weight].nmac
            factor[weight] = self.weights[weight].factor
                
        # alloc = branch_and_bound(self.params['narray'], nmac, factor, self.mac_per_array_layer, self.params)
        alloc = array_allocation(self.params['narray'], nmac, factor, self.mac_per_array_layer, self.params)
        assert (np.sum(alloc) <= self.params['narray'])
        # assert (np.sum(alloc) == 2 ** 14)
        print ("%d / %d" % (np.sum(alloc), self.params['narray']))

        for weight in range(len(self.weights)):
            layer_alloc = alloc[weight] // self.weights[weight].factor
            self.weights[weight].set_layer_alloc(layer_alloc)

    def set_block_alloc(self):
        nmac = np.zeros(shape=self.nblock, dtype=np.int32)
        factor = np.zeros(shape=self.nblock, dtype=np.int32)
        block = 0
        for weight in range(self.nweight):
            nwl, _, nbl, _ = np.shape(self.weights[weight].wb) 
            for wl in range(nwl):
                nmac[block] = self.weights[weight].nmac // nwl
                factor[block] = nbl
                block += 1
                
        # alloc = branch_and_bound(self.params['narray'], nmac, factor, self.mac_per_array_block, self.params)
        alloc = array_allocation(self.params['narray'], nmac, factor, self.mac_per_array_block, self.params)
        assert (np.sum(alloc) <= self.params['narray'])

        block1 = 0
        for weight in range(self.nweight):
            block2 = block1 + self.weights[weight].nwl
            block_alloc = np.array(alloc[block1:block2]) // self.weights[weight].nbl
            self.weights[weight].set_block_alloc(block_alloc)
            block1 = block2

#########################

class Layer:
    layer_id = 0
    weight_id = 0
    
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)

    def rpr(self):
        assert(False)
        
#########################

class Conv(Layer):
    def __init__(self, input_size, filter_size, pool, stride, pad1, pad2, params, weights, relu_flag=True):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        self.weight_id = Layer.weight_id
        Layer.weight_id += 1

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
        self.relu_flag = relu_flag

        assert (self.s == 1 or self.p == 1)
        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        maxval = pow(2, params['bpw'] - 1)
        minval = -1 * maxval

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['y']
        assert (np.all(self.w >= minval))
        assert (np.all(self.w <= maxval))
        # check shape
        assert(np.shape(self.w) == self.filter_size)
        assert(np.shape(self.b) == (self.fn,))
        assert(np.shape(self.q) == ())
        # cast as int
        self.w = self.w.astype(int)
        self.b = self.b.astype(int)
        self.q = self.q.astype(int)
        # q must be larger than 0
        # assert(self.q > 0)
        
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
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        y_min = np.min(y - y_ref)
        y_max = np.max(y - y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        # assert (self.s == 1)
        
        # print ('y_mean', y_mean, 'y_std', y_std)
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        # results = {}
        results['id']    = self.weight_id
        results['nmac']  = self.nmac
        results['std']   = y_std
        results['mean']  = y_mean
        
        nwl, _, nbl, _ = np.shape(self.wb)
        
        if self.params['alloc'] == 'block': 
            # the sum here is confusing, since for layer 1, block performs better with less arrays.
            # but it actually makes sense.
            results['array'] = np.sum(self.block_alloc) * nbl
            print ('alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall']))
                    
        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            print ('alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall']))

        ########################

        if self.relu_flag:
            y = relu(y)

        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.floor(y)
        y = np.clip(y, -128, 127)
        
        ########################

        return y, [results]
        
    def set_block_alloc(self, block_alloc):
        self.block_alloc = block_alloc

    def set_layer_alloc(self, layer_alloc):
        self.layer_alloc = layer_alloc

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
        
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(npatch, self.params['wl'] - (nrow % self.params['wl']), self.params['bpa']))
            patches = np.concatenate((patches, zeros), axis=1)
            
        patches = np.reshape(patches, (npatch, -1, self.params['wl'], self.params['bpa']))
        
        #########################
        
        if   self.params['alloc'] == 'block': alloc = self.block_alloc
        elif self.params['alloc'] == 'layer': alloc = self.layer_alloc
        
        y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], alloc, self.params)
        y = np.reshape(y, (yh, yw, self.fn))
        
        # we shud move this into forward, do it after the y - y_ref. 
        assert(np.all(np.absolute(y) < 2 ** 23))

        '''
        if self.relu_flag:
            y = relu(y)

        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.floor(y)
        y = np.clip(y, -128, 127)
        '''
        
        #########################
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        results = {}
        results['adc']   = metrics[0:8]
        results['cycle'] = metrics[8]
        results['ron']   = metrics[9]
        results['roff']  = metrics[10]
        results['wl']    = metrics[11]
        results['stall'] = metrics[12]
        results['block_cycle'] = metrics[13:]
        results['density'] = np.count_nonzero(patches) / np.prod(np.shape(patches)) * (self.params['wl'] / min(self.fh * self.fw * self.fc, self.params['wl']))
        results['block_density'] = np.count_nonzero(patches, axis=(0,2,3)) / (npatch * self.params['wl'] * self.params['bpa'])
        
        #########################
        
        return y, results
        
    def cut(self):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        w_offset = np.copy(self.w) + self.params['offset']
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
        # wb = np.transpose(wb, (0, 1, 3, 2))
        wb = np.reshape(wb, (nwl, self.params['wl'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['wl'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['wl'], -1, self.params['bl']))

        ########################

        return wb
        
    def weights(self):
        return [self]
        
#########################

class Block1(Layer):
    def __init__(self, input_size, filter_size, stride, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.f1, self.f2 = filter_size
        self.s = stride
        
        self.params = params.copy()
        self.params['var'] = lut_var(params['sigma'], 32)

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
        self.params['var'] = lut_var(params['sigma'], 32)

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

class MaxPool(Layer):
    def __init__(self, input_size, kernel_size, stride, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        
        self.input_size = input_size
        self.k = kernel_size
        self.s = stride

    def forward(self, x):
        y = avg_pool(x, self.k)
        y = np.clip(np.floor(y), -128, 127)
        return y, []

    def nblock(self):
        return 0

    def set_block_alloc(self, block_alloc):
        pass

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return []

#############










        
        
        
