
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes
lib = ctypes.cdll.LoadLibrary('./pim.so')
lib.pim.restype = ctypes.c_int

###########################

def cconv(x, f, b, q, pool, stride, pad1, pad2, params):
    assert (stride == 1)

    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    y = np.zeros(shape=(Ho, Wo, Co))
    
    ##################################################
    
    lut_rpr = params['rpr']
    lut_var = params['var']

    ##################################################

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    patches = []
    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            patches.append(patch)
            
    ##################################################

    patches = np.stack(patches, axis=0)
    pb = []
    for xb in range(params['bpa']):
        pb.append(np.bitwise_and(np.right_shift(patches.astype(int), xb), 1))
    
    patches = np.stack(pb, axis=-1)
    npatch, nrow, nbit = np.shape(patches)
    
    if (nrow % params['wl']):
        zeros = np.zeros(shape=(npatch, params['wl'] - (nrow % params['wl']), params['bpa']))
        patches = np.concatenate((patches, zeros), axis=1)
        
    patches = np.reshape(patches, (npatch, -1, params['wl'], params['bpa']))

    ##################################################
    
    f = f + params['offset']

    f = np.reshape(f, (Fh * Fw * Ci, Co))
    fb = []
    for wb in range(params['bpw']):
        fb.append(np.bitwise_and(np.right_shift(f.astype(int), wb), 1))
        
    f = np.stack(fb, axis=-1)
    
    nrow, ncol, nbit = np.shape(f)
    if (nrow % params['wl']):
        zeros = np.zeros(shape=(params['wl'] - (nrow % params['wl']), ncol, nbit))
        f = np.concatenate((f, zeros), axis=0)

    nrow, ncol, nbit = np.shape(f)
    f = np.reshape(f, (-1, params['wl'], ncol, nbit))

    nwl, wl, ncol, nbit = np.shape(f)
    f = np.transpose(f, (0, 1, 3, 2))
    f = np.reshape(f, (nwl, params['wl'], nbit * ncol))
    
    nwl, wl, ncol = np.shape(f)
    if (ncol % params['bl']):
        zeros = np.zeros(shape=(nwl, params['wl'], params['bl'] - (ncol % params['bl'])))
        f = np.concatenate((f, zeros), axis=2)

    f = np.reshape(f, (nwl, params['wl'], -1, params['bl']))
    
    ##################################################
    
    y, metrics = pim(patches, f, (Ho * Wo, Co), lut_var, lut_rpr, params)
    y = np.reshape(y, (Ho, Wo, Co))
    
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = relu(y)
    y = avg_pool(y, pool)
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)

    return y, metrics

##################################################

def pim(x, w, y_shape, lut_var, lut_rpr, ndup, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)

    # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl, stall
    metrics = np.zeros(shape=13)
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    lut_var = np.ascontiguousarray(lut_var, np.int32)
    lut_rpr = np.ascontiguousarray(lut_rpr, np.int32)
    metrics = np.ascontiguousarray(metrics, np.int32)

    # print (lut_rpr)
    # print (np.array(lut_rpr.ctypes.strides))
    # assert (False)

    psum = lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(lut_var.ctypes.data), 
    ctypes.c_void_p(lut_rpr.ctypes.data), 
    ctypes.c_void_p(metrics.ctypes.data), 
    ctypes.c_int(params['adc']),
    ctypes.c_int(params['skip']),
    ctypes.c_int(nrow),
    ctypes.c_int(ndup),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))
    
    return y, metrics
    
    
    
