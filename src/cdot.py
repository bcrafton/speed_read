
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes

pim_lib = ctypes.cdll.LoadLibrary('./pim.so')
pim_lib.pim.restype = ctypes.c_int

###########################

def pim(x, w, y_shape, lut_var, lut_rpr, alloc, adc_state, adc_thresh, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)

    # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl, stall, block_cycles[nwl]
    metrics_len = 13 + nwl
    metrics = np.zeros(shape=metrics_len)
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    lut_var = np.ascontiguousarray(lut_var, np.float32)
    lut_rpr = np.ascontiguousarray(lut_rpr, np.int32)
    adc_state = np.ascontiguousarray(adc_state, np.float32)
    adc_thresh = np.ascontiguousarray(adc_thresh, np.float32)
    metrics = np.ascontiguousarray(metrics, np.int64)

    lut_bias = np.zeros(shape=64)
    lut_bias = np.ascontiguousarray(lut_bias, np.int32)

    ########

    if params['alloc'] == 'block':
        nblock = np.sum(alloc)    
        block_map = np.zeros(shape=nblock)
        block = 0
        for i in range(nwl):
            for j in range(alloc[i]):
                block_map[block] = i
                block += 1
        
        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 0
        
    ########

    elif params['alloc'] == 'layer':
        nblock = alloc * nwl
        block_map = np.zeros(shape=(alloc, nwl))
        for i in range(alloc):
            for j in range(nwl):
                block_map[i][j] = j
        
        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 1
    
    ########
    
    # print (adc_state)
    # print (adc_thresh)
    # print (lut_rpr)

    psum = pim_lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(lut_var.ctypes.data), 
    ctypes.c_void_p(lut_rpr.ctypes.data), 
    ctypes.c_void_p(lut_bias.ctypes.data),
    ctypes.c_void_p(metrics.ctypes.data), 
    ctypes.c_void_p(block_map.ctypes.data),
    ctypes.c_void_p(adc_state.ctypes.data), 
    ctypes.c_void_p(adc_thresh.ctypes.data), 
    ctypes.c_int(params['adc']),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(params['skip']),
    ctypes.c_int(nrow),
    ctypes.c_int(nblock),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl),
    ctypes.c_int(sync),
    ctypes.c_int(1))
    
    ########
    
    return y, metrics
    
###########################

# copying this from cc_update1
def pim_dyn(x, w, y_shape, lut_var, lut_rpr, alloc, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)

    # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl, stall, block_cycles[nwl]
    metrics_len = 13 + nwl
    metrics = np.zeros(shape=metrics_len)
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    lut_var = np.ascontiguousarray(lut_var, np.float32)
    lut_rpr = np.ascontiguousarray(lut_rpr, np.int32)
    metrics = np.ascontiguousarray(metrics, np.int64)
    
    lut_bias = np.zeros(shape=64)
    lut_bias = np.ascontiguousarray(lut_bias, np.int32)
    
    # self.adc_state = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
    # self.adc_thresh = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
    adc_state = np.zeros(shape=(64, 9))
    adc_thresh = np.zeros(shape=(64, 9))
    
    adc_state = np.ascontiguousarray(adc_state, np.float32)
    adc_thresh = np.ascontiguousarray(adc_thresh, np.float32)

    ########

    if params['alloc'] == 'block':
        nblock = np.sum(alloc)    
        block_map = np.zeros(shape=nblock)
        block = 0
        for i in range(nwl):
            for j in range(alloc[i]):
                block_map[block] = i
                block += 1
        
        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 0

    elif params['alloc'] == 'layer':
        nblock = alloc * nwl
        block_map = np.zeros(shape=(alloc, nwl))
        for i in range(alloc):
            for j in range(nwl):
                block_map[i][j] = j
        
        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 1
    
    ########

    psum = pim_lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(lut_var.ctypes.data), 
    ctypes.c_void_p(lut_rpr.ctypes.data), 
    ctypes.c_void_p(lut_bias.ctypes.data),
    ctypes.c_void_p(metrics.ctypes.data), 
    ctypes.c_void_p(block_map.ctypes.data),
    ctypes.c_void_p(adc_state.ctypes.data), 
    ctypes.c_void_p(adc_thresh.ctypes.data), 
    ctypes.c_int(params['adc']),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(params['skip']),
    ctypes.c_int(nrow),
    ctypes.c_int(nblock),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl),
    ctypes.c_int(sync),
    ctypes.c_int(0))
    
    return y, metrics

###########################

def pim_static(x, w, y_shape, lut_var, lut_rpr, alloc, lut_bias, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape

    y = np.zeros(shape=y_shape)

    # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl, stall, block_cycles[nwl]
    metrics_len = 13 + nwl + (nwl * nbl)
    metrics = np.zeros(shape=metrics_len)

    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    lut_var = np.ascontiguousarray(lut_var, np.float32)
    lut_rpr = np.ascontiguousarray(lut_rpr, np.int32)
    metrics = np.ascontiguousarray(metrics, np.int64)

    lut_bias = np.ascontiguousarray(lut_bias, np.int32)

    # self.adc_state = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
    # self.adc_thresh = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
    adc_state = np.zeros(shape=(64, 9))
    adc_thresh = np.zeros(shape=(64, 9))

    adc_state = np.ascontiguousarray(adc_state, np.float32)
    adc_thresh = np.ascontiguousarray(adc_thresh, np.float32)

    ########

    if params['alloc'] == 'block':
        nblock = np.sum(alloc)
        block_map = np.zeros(shape=nblock)
        block = 0
        for i in range(nwl):
            for j in range(alloc[i]):
                block_map[block] = i
                block += 1

        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 0

    elif params['alloc'] == 'layer':
        nblock = alloc * nwl
        block_map = np.zeros(shape=(alloc, nwl))
        for i in range(alloc):
            for j in range(nwl):
                block_map[i][j] = j

        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        sync = 1

    ########

    psum = pim_lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(lut_var.ctypes.data), 
    ctypes.c_void_p(lut_rpr.ctypes.data), 
    ctypes.c_void_p(lut_bias.ctypes.data),
    ctypes.c_void_p(metrics.ctypes.data), 
    ctypes.c_void_p(block_map.ctypes.data),
    ctypes.c_void_p(adc_state.ctypes.data), 
    ctypes.c_void_p(adc_thresh.ctypes.data), 
    ctypes.c_int(params['adc']),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(params['skip']),
    ctypes.c_int(params['ABFT']),
    ctypes.c_int(params['ABFT_XB']),
    ctypes.c_int(params['ABFT_ADC']),
    ctypes.c_int(nrow),
    ctypes.c_int(nblock),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl),
    ctypes.c_int(sync),
    ctypes.c_int(2))
    
    return y, metrics
    
