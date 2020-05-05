
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes

pim_lib = ctypes.cdll.LoadLibrary('./pim.so')
pim_lib.pim.restype = ctypes.c_int

pim_sync_lib = ctypes.cdll.LoadLibrary('./pim_sync.so')
pim_sync_lib.pim.restype = ctypes.c_int

###########################

def pim(x, w, y_shape, lut_var, lut_rpr, alloc, adc_thresh, params):
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
    adc_thresh = np.ascontiguousarray(adc_thresh, np.float32)
    metrics = np.ascontiguousarray(metrics, np.int32)

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
        
    ########

    elif params['alloc'] == 'layer':
        nblock = alloc * nwl
        block_map = np.zeros(shape=(alloc, nwl))
        for i in range(alloc):
            for j in range(nwl):
                block_map[i][j] = j
        
        block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
    
    ########

    if params['alloc'] == 'block':
        psum = pim_lib.pim(
        ctypes.c_void_p(x.ctypes.data), 
        ctypes.c_void_p(w.ctypes.data), 
        ctypes.c_void_p(y.ctypes.data), 
        ctypes.c_void_p(lut_var.ctypes.data), 
        ctypes.c_void_p(lut_rpr.ctypes.data), 
        ctypes.c_void_p(metrics.ctypes.data), 
        ctypes.c_void_p(block_map.ctypes.data), 
        ctypes.c_void_p(adc_thresh.ctypes.data), 
        ctypes.c_int(params['adc']),
        ctypes.c_int(params['skip']),
        ctypes.c_int(nrow),
        ctypes.c_int(nblock),
        ctypes.c_int(ncol),
        ctypes.c_int(nwl),
        ctypes.c_int(nbl),
        ctypes.c_int(wl),
        ctypes.c_int(bl))
    
    ########
    
    if params['alloc'] == 'layer':
        psum = pim_sync_lib.pim(
        ctypes.c_void_p(x.ctypes.data), 
        ctypes.c_void_p(w.ctypes.data), 
        ctypes.c_void_p(y.ctypes.data), 
        ctypes.c_void_p(lut_var.ctypes.data), 
        ctypes.c_void_p(lut_rpr.ctypes.data), 
        ctypes.c_void_p(metrics.ctypes.data), 
        ctypes.c_int(params['adc']),
        ctypes.c_int(params['skip']),
        ctypes.c_int(nrow),
        ctypes.c_int(alloc),
        ctypes.c_int(ncol),
        ctypes.c_int(nwl),
        ctypes.c_int(nbl),
        ctypes.c_int(wl),
        ctypes.c_int(bl))
    
    ########
    
    return y, metrics
    
    
    
