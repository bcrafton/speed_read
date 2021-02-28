
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

import sys, os, psutil

#########################

class Linear(Layer):
    def __init__(self, size, params, weights):
        self.params = params.copy()

        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        self.weight_id = Layer.weight_id
        Layer.weight_id += 1

        self.size = size
        self.input_size, self.output_size = self.size

        remainder = self.output_size % (self.params['bl'] // self.params['bpw'])
        self.output_size_pad = self.output_size
        if remainder: self.output_size_pad += (self.params['bl'] // self.params['bpw']) - remainder

        self.w = weights['w'].astype(np.int8)
        self.q = 1. / weights['sx']

        assert(np.shape(self.w) == self.size)
        maxval = pow(2, self.params['bpw'] - 1)
        minval = -1 * maxval
        assert (np.all(self.w >= minval))
        assert (np.all(self.w <= maxval))

        wb = self.transform_weights().astype(np.int8)
        self.w_shape = np.shape(wb)
        nwl, _, nbl, _ = self.w_shape
        self.wb = np.packbits(wb)

        self.params['nwl'] = nwl
        self.params['nbl'] = nbl
        self.params['total_array'] = nwl * nbl
        self.params['total_mac'] = self.input_size * self.output_size

    def init(self, params, table):
        self.params.update(params)
        self.params['var'] = lut_var(params['sigma'], self.params['max_rpr'])
        table[self.weight_id] = self

        if self.params['rpr_alloc'] == 'centroids':
            assert (False)
        elif self.params['rpr_alloc'] == 'dynamic':
            assert (False)
        elif self.params['rpr_alloc'] == 'static':
            '''
            self.params['rpr'], self.lut_bias = static_rpr(low=1, 
                                                           high=self.params['max_rpr'], 
                                                           params=self.params, 
                                                           adc_count=self.adc_count, 
                                                           row_count=self.row_count, 
                                                           sat_count=self.sat_count, 
                                                           nrow=self.input_size, 
                                                           q=self.q, 
                                                           ratio=self.ratio)
            '''
            self.params['rpr'] = np.ones(shape=(8, 8)).astype(np.int32) * 8
            self.lut_bias = np.zeros(shape=(8, 8)).astype(np.int32)
        else:
            assert (False)

    def get(self, arg):
        return self.params[arg]

    def set(self, arg, value):
        self.params[arg] = value

    def set_profile_adc(self, counts):
        # it makes no sense to invert [parameter, layer_id]
        # we should set it back
        # only need change [linear.py, bert.py]
        self.adc_count = counts['adc'][self.layer_id]
        self.row_count = counts['row'][self.layer_id]
        self.sat_count = counts['sat'][self.layer_id]
        self.ratio = counts['ratio'][self.layer_id]

    def profile_adc(self, x, counters):
        rpr_low = 1
        rpr_high = self.params['max_rpr']        
        patches = self.transform_inputs(x).astype(np.int8)
        npatch, nwl, wl, xb = np.shape(patches)

        rpr  = np.arange(rpr_low, rpr_high + 1)
        nrow = np.sum(patches, axis=2)
        nrow = nrow.reshape(npatch, nwl, xb, 1)
        nrow = np.ceil(nrow / rpr)
        nrow = np.clip(nrow, 1, np.inf)
        nrow = np.sum(nrow, axis=1)
        nrow = np.mean(nrow, axis=0)
        
        y_ref = x @ self.w
        ratio = np.count_nonzero(y_ref) / np.prod(np.shape(y_ref))
        
        x_shape = np.shape(patches)
        patches = np.packbits(patches)
        
        # it makes no sense to invert [parameter, layer_id]
        # we should set it back
        # only need change [linear.py, bert.py]
        counters['adc'].update({self.layer_id: (x_shape, patches, self.w_shape, self.wb, (npatch, self.output_size_pad), rpr_low, rpr_high, self.params)})
        counters['ratio'].update({self.layer_id: ratio})
        counters['row'].update({self.layer_id: nrow})
        return y_ref
        
    def weights(self):
        assert (False)
        return [self]

    def forward(self, x, results):
        assert (self.weight_id not in results.keys())
        results[self.weight_id] = {}
        ########################
        word_size, vector_size = np.shape(x)
        assert (vector_size == np.shape(self.w)[0])
        ########################
        y = self.conv(x, results)
        y_ref = dot_ref(x=x, w=self.w, b=None, q=None)
        ########################
        mean = np.mean(y - y_ref)
        error = np.mean(np.absolute(y - y_ref))
        results[self.weight_id]['cim_mean'] = mean
        results[self.weight_id]['cim_error'] = error
        ########################
        y_min = np.min(y_ref)
        y_max = np.max(y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        y_error = np.mean(np.absolute(y - y_ref))
        ########################
        results[self.weight_id]['id']        = self.weight_id
        # results[self.weight_id]['nmac']      = self.params['total_mac']
        results[self.weight_id]['nmac']      = self.input_size * word_size * self.output_size
        results[self.weight_id]['nwl']       = self.params['nwl']
        results[self.weight_id]['nbl']       = self.params['nbl']
        results[self.weight_id]['std']       = y_std
        results[self.weight_id]['mean']      = y_mean
        results[self.weight_id]['error']     = y_error
        results[self.weight_id]['duplicate'] = self.params['duplicate']
        ########################
        p = '%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d mean: %0.3f error: %0.3f q: %0.3f' % (
            self.layer_id, 
            np.sum(self.params['duplicate']), 
            self.params['nbl'], 
            self.params['nbl'] * np.sum(self.params['duplicate']), 
            results[self.weight_id]['nmac'], 
            results[self.weight_id]['cycle'], 
            results[self.weight_id]['stall'], 
            y_mean, 
            y_error, 
            self.q)
        print (p)
        ########################
        return y_ref
        
    def conv(self, x, results):
        xb = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(xb)
        #########################
        if self.params['rpr_alloc'] == 'centroids':
            assert (False)
        elif self.params['rpr_alloc'] == 'dynamic':
            assert (False)
        elif self.params['rpr_alloc'] == 'static':
            # think we want to pass a bias table
            wb = np.unpackbits(self.wb).reshape(self.w_shape)
            y, metrics = pim_static(xb, wb, (npatch, self.output_size_pad), self.params['var'], self.params['rpr'], self.params['duplicate'], self.lut_bias, self.params)
            y = np.reshape(y, (npatch, self.output_size_pad))[:, 0:self.output_size]
            # metrics = np.random.randint(low=1000, high=10000, size=np.shape(metrics))
        else:
            assert (False)
        #########################
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        results[self.weight_id]['adc']   = metrics[0:8]
        results[self.weight_id]['cycle'] = metrics[8]
        results[self.weight_id]['ron']   = metrics[9]
        results[self.weight_id]['roff']  = metrics[10]
        results[self.weight_id]['wl']    = metrics[11]
        results[self.weight_id]['stall'] = metrics[12]
        results[self.weight_id]['block_cycle'] = metrics[13:]
        results[self.weight_id]['density'] = np.count_nonzero(xb) / np.prod(np.shape(xb)) * (self.params['wl'] / min(self.input_size, self.params['wl']))
        results[self.weight_id]['block_density'] = np.count_nonzero(xb, axis=(0,2,3)) / (npatch * self.params['wl'] * self.params['bpa'])
        #########################
        return y
        
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
        w_offset = self.w + self.params['offset']
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


        
        
        
