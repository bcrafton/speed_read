
import math
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
from kmeans import kmeans

from conv_utils import *
from cdot import *
from dot_ref import *
from var import *

from layers import *
from cprofile import profile
from dynamic_rpr import dynamic_rpr
from static_rpr import static_rpr
from kmeans_rpr import kmeans_rpr
from cim import cim

#########################

class Conv(Layer):
    def __init__(self, input_size, filter_size, pool, stride, pad1, pad2, params, weights, relu_flag=True, quantize_flag=True):
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
        self.quantize_flag = quantize_flag

        self.params = params.copy()

        assert (self.s == 1 or self.p == 1)

        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        self.yw = self.yh

        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['q']
        # check shape
        assert(np.shape(self.w) == self.filter_size)
        assert(np.shape(self.b) == (self.fn,))
        assert(np.shape(self.q) == ())
        # cast as int
        assert(np.max(self.w) < 128 and np.min(self.w) >= -128)
        self.w = self.w.astype(np.int8)
        self.b = self.b.astype(np.float32)
        self.q = self.q.astype(np.float32)
        # q must be larger than 0
        if self.quantize_flag:
            assert(self.q > 0)

        maxval = pow(2, self.params['bpw'] - 1)
        minval = -1 * maxval

        assert (np.all(self.w >= minval))
        assert (np.all(self.w <= maxval))

        self.wb = self.transform_weights()
        nwl, _, nbl, _ = np.shape(self.wb) 
        self.factor = nwl * nbl
        self.nwl = nwl
        self.nbl = nbl

        #########################

    def init(self, params):
        self.params.update(params)
        self.params['var'] = lut_var(params['lrs'], params['hrs'], self.params['max_rpr'])

        if self.params['rpr_alloc'] == 'centroids':
            pass
        elif self.params['rpr_alloc'] == 'dynamic':
            pass
        elif self.params['rpr_alloc'] == 'static':
            self.params['rpr'], self.params['step'], self.error, self.mean = static_rpr(self.layer_id, self.params, self.q)
        else:
            assert (False)

    def profile_adc(self, x):
        # x
        patches = self.transform_inputs(x)
        # w 
        w = self.wb
        # y
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y_ref = self.act(y_ref, quantize_flag=self.quantize_flag)
        # y_shape
        y_shape = (self.yh * self.yw, self.c)

        return y_ref, [(self.layer_id, patches, self.wb, y_ref, y_shape, self.params)]

    def set_block_alloc(self, block_alloc):
        self.block_alloc = block_alloc

    def set_layer_alloc(self, layer_alloc):
        self.layer_alloc = layer_alloc
        
    def weights(self):
        return [self]

    def act(self, y, quantize_flag):
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        assert(self.p == 1)
        # y = avg_pool(y, self.p)
        if quantize_flag:
            y = y / self.q
            y = np.around(y)
            y = np.clip(y, -128, 127)
        return y

    def forward(self, x, x_ref, profile=False):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x_ref, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        mean = np.mean(y - y_ref)
        error = np.mean(np.absolute(y - y_ref))
        std = np.std(y - y_ref)
        results['cim_mean'] = mean
        results['cim_error'] = error
        results['cim_std'] = std

        z = self.act(y, quantize_flag=True)
        z_ref = self.act(y_ref, quantize_flag=True)

        # nonzero = np.count_nonzero(z_ref) / np.prod(np.shape(z_ref))
        # print (nonzero)

        z_min = np.min(z_ref)
        z_max = np.max(z_ref)
        z_mean = np.mean(z - z_ref)
        z_std = np.std(z - z_ref)
        z_error = np.mean(np.absolute(z - z_ref))

        # print (self.error, self.mean)
        # print (error * self.ratio / self.q, mean * self.ratio / self.q)
        # print (error / self.q * self.ratio)
        # print (self.params['rpr'])

        results['id']       = self.weight_id
        results['layer_id'] = self.layer_id
        results['nmac']     = self.nmac
        results['std']      = z_std
        results['mean']     = z_mean
        results['error']    = z_error

        nwl, _, nbl, _ = np.shape(self.wb)

        if self.params['alloc'] == 'block':
            results['array'] = np.sum(self.block_alloc) * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d mean: %0.3f std: %0.3f error: %0.3f' % 
              (self.layer_id, np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall'], z_mean, z_std, z_error))

        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d mean: %0.2f error: %0.2f' % 
              (self.layer_id, self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall'], z_mean, z_error))

        ########################

        y = self.act(y, quantize_flag=self.quantize_flag)
        y_ref = self.act(y_ref, quantize_flag=self.quantize_flag)

        ########################

        # accuracy: y_ref = y
        # perf:     y = y_ref
        # error:    None

        y = y_ref
        # y_ref = y

        ########################

        return y, y_ref, [results]
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        patches = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(patches)
        
        #########################
        
        if   self.params['alloc'] == 'block': alloc = self.block_alloc
        elif self.params['alloc'] == 'layer': alloc = self.layer_alloc
        
        if self.params['rpr_alloc'] == 'centroids':
            assert (False)
        elif self.params['rpr_alloc'] == 'dynamic':
            assert (False)
        elif self.params['rpr_alloc'] == 'static':
            # think we want to pass a bias table
            y, metrics = pim_static(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], self.params['step'], alloc, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
        else:
            assert (False)
        
        # we shud move this into forward, do it after the y - y_ref. 
        assert(np.all(np.absolute(y) < 2 ** 23))

        #########################
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        results = {}
        results['cycle']       = metrics[0]
        results['ron']         = metrics[1]
        results['roff']        = metrics[2]
        results['wl']          = metrics[3]
        results['stall']       = metrics[4]

        results['block_cycle'] = metrics[5:][:nwl]

        results['adc'] = metrics[5+nwl:]
        results['adc'] = np.reshape(results['adc'], (8, 8, nwl, self.params['adc'] + 1))

        results['density'] = np.count_nonzero(patches) / np.prod(np.shape(patches)) * (self.params['wl'] / min(self.fh * self.fw * self.fc, self.params['wl']))
        results['block_density'] = np.count_nonzero(patches, axis=(0,2,3)) / (npatch * self.params['wl'] * self.params['bpa'])

        results['block_alloc'] = self.block_alloc
        results['block_size'] = self.nbl
        results['rpr'] = self.params['rpr']
        results['step'] = self.params['step']

        #########################
        
        # y, metrics = cim(patches, self.wb, self.params)
        # y = np.reshape(y, (yh, yw, self.fn))

        #########################

        return y, results
        
    def transform_inputs(self, x):
    
        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        #########################
        
        x = np.pad(array=x, pad_width=[[self.p1,self.p2], [self.p1,self.p2], [0,0]], mode='constant')
        patches = []
        for h in range(yh):
            for w in range(yw):
                patch = x[h*self.s:(h*self.s+self.fh), w*self.s:(w*self.s+self.fw), :]
                patch = np.reshape(patch, self.fh * self.fw * self.fc)
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
        
        return patches
        
    def transform_weights(self):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        w_matrix = np.reshape(np.copy(self.w), (self.fh * self.fw * self.fc, self.fn))
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



        
        
        
