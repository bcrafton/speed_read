
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
from rpr import rpr as dynamic_rpr
from static_rpr import static_rpr

from kmeans_config import KmeansConfig

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

        self.params = params.copy()

        assert (self.s == 1 or self.p == 1)

        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        self.yw = self.yh

        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['y']
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
        
        self.params['var'] = lut_var(params['sigma'], 64)

        if self.params['rpr_alloc'] == 'centroids':
            cfg = KmeansConfig(low=1, high=64, params=self.params, adc_count=self.adc_count, row_count=self.row_count, nrow=self.fh * self.fw * self.fc, q=self.q)
            self.params['rpr'], self.adc_state, self.adc_thresh = cfg.rpr()

        elif self.params['rpr_alloc'] == 'dynamic':
            ## TODO: cant this be "self.wb" and cant we throw it in a different function ??
            '''
            w_offset = self.w + self.params['offset']
            wb = []
            for bit in range(self.params['bpw']):
                wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
            wb = np.stack(wb, axis=-1)

            wb_cols = np.reshape(wb, (self.fh * self.fw * self.fc, self.fn, self.params['bpw']))
            col_density = np.mean(wb_cols, axis=0)

            nrow = self.fh * self.fw * self.fc
            p = np.max(col_density, axis=0)
            self.params['rpr'] = dynamic_rpr(nrow=nrow, p=p, q=self.q, params=self.params)
            # print (self.params['rpr'])
            '''
            self.params['rpr'], _ = static_rpr(low=1, high=16, params=self.params, adc_count=self.adc_count, row_count=self.row_count, nrow=self.fh * self.fw * self.fc, q=self.q)

        elif self.params['rpr_alloc'] == 'static':
            self.params['rpr'], self.lut_bias = static_rpr(low=1, high=16, params=self.params, adc_count=self.adc_count, row_count=self.row_count, nrow=self.fh * self.fw * self.fc, q=self.q)
            self.lut_bias = self.lut_bias * 256
            self.lut_bias = self.lut_bias.astype(np.int32)
        else:
            assert (False)

    def set_profile_adc(self, counts):
        self.adc_count = counts[self.layer_id]['adc']
        self.row_count = counts[self.layer_id]['row']

    def profile_adc(self, x):
        rpr_low = 1
        rpr_high = 64
        patches = self.transform_inputs(x)
        _, self.adc_count, self.row_count = profile(patches, self.wb, (self.yh * self.yw, self.fn), rpr_low, rpr_high, self.params)
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y_ref = self.act(y_ref)
        return y_ref, {self.layer_id: {'adc': self.adc_count, 'row': self.row_count}}

    def set_block_alloc(self, block_alloc):
        self.block_alloc = block_alloc

    def set_layer_alloc(self, layer_alloc):
        self.layer_alloc = layer_alloc
        
    def weights(self):
        return [self]

    def act(self, y):
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.round(y)
        y = np.clip(y, -128, 127)
        return y

    def forward(self, x, profile=False):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        y = self.act(y)
        y_ref = self.act(y_ref)

        y_min = np.min(y_ref)
        y_max = np.max(y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        y_error = np.mean(np.absolute(y - y_ref))
        # assert (self.s == 1)
        
        print ('y_mean', y_mean, 'y_error', y_error, 'y_max', y_max, 'y_min', y_min)
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        # results = {}
        results['id']    = self.weight_id
        results['nmac']  = self.nmac
        results['std']   = y_std
        results['mean']  = y_mean
        results['error'] = y_error
        
        nwl, _, nbl, _ = np.shape(self.wb)
        
        if self.params['alloc'] == 'block': 
            # the sum here is confusing, since for layer 1, block performs better with less arrays.
            # but it actually makes sense.
            results['array'] = np.sum(self.block_alloc) * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall']))
                    
        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall']))

        ########################

        y = y_ref
        # assert (y_std <= 0)
        
        '''
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.round(y)
        y = np.clip(y, -128, 127)
        '''
        ########################

        return y, [results]
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        patches = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(patches)
        
        #########################
        
        if   self.params['alloc'] == 'block': alloc = self.block_alloc
        elif self.params['alloc'] == 'layer': alloc = self.layer_alloc
        
        if self.params['rpr_alloc'] == 'centroids':
            y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], alloc, self.adc_state, self.adc_thresh, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
            y = y / 4
        elif self.params['rpr_alloc'] == 'dynamic':
            # want to pass some table to C instead of computing stuff inside.
            y, metrics = pim_dyn(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], alloc, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
        elif self.params['rpr_alloc'] == 'static':
            # think we want to pass a bias table
            y, metrics = pim_static(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], alloc, self.lut_bias, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
        else:
            assert (False)
        
        # we shud move this into forward, do it after the y - y_ref. 
        assert(np.all(np.absolute(y) < 2 ** 23))

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



        
        
        
