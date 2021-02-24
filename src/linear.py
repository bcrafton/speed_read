
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

#########################

class Linear(Layer):
    def __init__(self, size, params, weights):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        self.weight_id = Layer.weight_id
        Layer.weight_id += 1

        self.size = size
        # TODO: include word size ... [128, 768]
        self.input_size, self.output_size = self.size

        self.params = params.copy()

        remainder = self.output_size % (self.params['bl'] // self.params['bpw'])
        self.output_size_pad = self.output_size
        if remainder:
            self.output_size_pad += (self.params['bl'] // self.params['bpw']) - remainder

        # TODO: include word size ... [128, 768]
        self.nmac = self.input_size * self.output_size

        # self.w, self.b, self.q = weights['w'], weights['b'], weights['q']
        self.w = weights['w']
        # check shape
        assert(np.shape(self.w) == self.size)
        # assert(np.shape(self.b) == (self.output_size,))
        # assert(np.shape(self.q) == ())
        # cast as int
        self.w = self.w.astype(int)
        # self.b = self.b.astype(int)
        # self.q = self.q.astype(int)
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
        
        self.params['var'] = lut_var(params['sigma'], self.params['max_rpr'])

        if self.params['rpr_alloc'] == 'centroids':
            assert (False)

        elif self.params['rpr_alloc'] == 'dynamic':
            self.params['rpr'], _ = static_rpr(low=1, high=self.params['max_rpr'], params=self.params, adc_count=self.adc_count, row_count=self.row_count, sat_count=self.sat_count, nrow=self.input_size, q=self.q)

        elif self.params['rpr_alloc'] == 'static':
            self.params['rpr'], self.lut_bias = static_rpr(low=1, high=self.params['max_rpr'], params=self.params, adc_count=self.adc_count, row_count=self.row_count, sat_count=self.sat_count, nrow=self.input_size, q=self.q)
            self.lut_bias = self.lut_bias * 256
            self.lut_bias = self.lut_bias
        else:
            self.params['rpr'] = np.ones(shape=(8, 8)).astype(np.int32) * 8
            self.lut_bias = np.zeros(shape=(8, 8)).astype(np.int32)

        self.block_alloc = np.ones(shape=self.nwl).astype(np.int32)

    def set_profile_adc(self, counts):
        self.adc_count = counts[self.layer_id]['adc']
        self.row_count = counts[self.layer_id]['row']
        self.sat_count = counts[self.layer_id]['sat']

    def profile_adc(self, x):
        rpr_low = 1
        rpr_high = self.params['max_rpr']
        x = np.reshape(x, self.input_size)
        patches = self.transform_inputs(x)
        # _, self.adc_count, self.row_count = profile(patches, self.wb, (1, self.output_size), rpr_low, rpr_high, self.params)
        y_ref = dot_ref(x=x, w=self.w, b=self.b, q=self.q)
        y_ref = self.act(y_ref)
        return y_ref, {self.layer_id: (patches, self.wb, (1, self.output_size_pad), rpr_low, rpr_high, self.params)}

    def set_block_alloc(self, block_alloc):
        self.block_alloc = block_alloc
        assert( np.sum(self.block_alloc) == self.nbl )

    def set_layer_alloc(self, layer_alloc):
        self.layer_alloc = layer_alloc
        assert( np.sum(self.layer_alloc) == 1 )
        
    def weights(self):
        return [self]

    def act(self, y):
        return y

    def forward(self, x, x_ref, profile=False):
        word_size, vector_size = np.shape(x)
        assert (vector_size == np.shape(self.w)[0])
        # y_ref = dot_ref(x=x_ref, w=self.w, b=self.b, q=self.q)
        y_ref = dot_ref(x=x_ref, w=self.w, b=None, q=None)
        y, results = self.conv(x=x)

        # print (y_ref.flatten()[0:10])
        # print (x.flatten()[0:10])
        sparse = np.count_nonzero(self.w) / np.prod(np.shape(self.w))
        # print (sparse)

        # idx = np.where((y - y_ref) != 0)
        # print (np.shape(y))
        # print (np.shape(y[0]))
        # print (np.shape(y_ref))
        # print (y_ref[idx])
        # print (y[idx])

        mean = np.mean(y - y_ref)
        error = np.mean(np.absolute(y - y_ref))
        results['cim_mean'] = mean
        results['cim_error'] = error
        '''
        y = self.act(y)
        y_ref = self.act(y_ref)
        '''
        y_min = np.min(y_ref)
        y_max = np.max(y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        y_error = np.mean(np.absolute(y - y_ref))
        # assert (self.s == 1)
        
        # print ('y_mean', y_mean, 'y_error', y_error, 'y_max', y_max, 'y_min', y_min)
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        # results = {}
        results['id']    = self.weight_id
        results['nmac']  = self.nmac
        results['std']   = y_std
        results['mean']  = y_mean
        results['error'] = y_error
        
        nwl, _, nbl, _ = np.shape(self.wb)
        
        if self.params['alloc'] == 'block':
            results['array'] = np.sum(self.block_alloc) * nbl
            # print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall']))
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d mean: %0.3f error: %0.3f' % 
              (self.layer_id, np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall'], y_mean, y_error))

        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            # print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall']))
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d mean: %0.2f error: %0.2f' % 
              (self.layer_id, self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall'], y_mean, y_error))

        ########################

        # y = y_ref
        # y_ref = y
        return y, y_ref, [results]
        
    def conv(self, x):
        
        xb = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(xb)
        #########################
        if   self.params['alloc'] == 'block': alloc = self.block_alloc
        elif self.params['alloc'] == 'layer': alloc = self.layer_alloc
        
        if self.params['rpr_alloc'] == 'centroids':
            # y, metrics = pim(xb, self.wb, (1, self.output_size_pad), self.params['var'], self.params['rpr'], alloc, self.adc_state, self.adc_thresh, self.params)
            # y = np.reshape(y, self.output_size_pad)[:self.output_size]
            # y = y / 4
            assert (False)
        elif self.params['rpr_alloc'] == 'dynamic':
            # want to pass some table to C instead of computing stuff inside.
            # y, metrics = pim_dyn(xb, self.wb, (1, self.output_size_pad), self.params['var'], self.params['rpr'], alloc, self.params)
            # y = np.reshape(y, self.output_size_pad)[:self.output_size]
            assert (False)
        elif self.params['rpr_alloc'] == 'static':
            # think we want to pass a bias table
            # y, metrics = pim_static(xb, self.wb, (1, self.output_size_pad), self.params['var'], self.params['rpr'], alloc, self.lut_bias, self.params)
            # y = np.reshape(y, self.output_size_pad)[:self.output_size]
            assert (False)
        else:
            y, metrics = pim_dyn(xb, self.wb, (npatch, self.output_size_pad), self.params['var'], self.params['rpr'], alloc, self.params)
            y = np.reshape(y, (npatch, self.output_size_pad))
        
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
        results['density'] = np.count_nonzero(xb) / np.prod(np.shape(xb)) * (self.params['wl'] / min(self.input_size, self.params['wl']))
        results['block_density'] = np.count_nonzero(xb, axis=(0,2,3)) / (npatch * self.params['wl'] * self.params['bpa'])
        #########################
        return y, results
        
    def transform_inputs(self, x):
        #########################

        xb = []
        for bit in range(self.params['bpa']):
            xb.append(np.bitwise_and(np.right_shift(x.astype(int), bit), 1))
        
        xb = np.stack(xb, axis=-1)
        nword, nrow, nbit = np.shape(xb)
        
        #########################
        
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(npatch, self.params['wl'] - (nrow % self.params['wl']), self.params['bpa']))
            xb = np.concatenate((xb, zeros), axis=1)
            
        xb = np.reshape(xb, (nword, -1, self.params['wl'], self.params['bpa']))

        #########################
        
        return xb
        
    def transform_weights(self):
        w_offset = np.copy(self.w) + self.params['offset']
        w_matrix = np.reshape(w_offset, (self.input_size, self.output_size))
        
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
        wb = np.reshape(wb, (nwl, self.params['wl'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['wl'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['wl'], -1, self.params['bl']))

        ########################

        return wb


        
        
        
