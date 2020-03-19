
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes
lib = ctypes.cdll.LoadLibrary('./pim.so')
lib.pim.restype = ctypes.c_int

###########################

def get_lut_var(var, rpr):
    lut = np.zeros(shape=(rpr + 1, 1000), dtype=np.int32)
    for s in range(1, rpr + 1):
        
        std = var * np.sqrt(s)
        
        p5 = norm.cdf( 5.5, 0, std) - norm.cdf( 4.5, 0, std)
        p4 = norm.cdf( 4.5, 0, std) - norm.cdf( 3.5, 0, std)
        p3 = norm.cdf( 3.5, 0, std) - norm.cdf( 2.5, 0, std)
        p2 = norm.cdf( 2.5, 0, std) - norm.cdf( 1.5, 0, std)
        p1 = norm.cdf( 1.5, 0, std) - norm.cdf( 0.5, 0, std)

        p5 = int(round(p5, 3) * 1000)
        p4 = int(round(p4, 3) * 1000)
        p3 = int(round(p3, 3) * 1000)
        p2 = int(round(p2, 3) * 1000)
        p1 = int(round(p1, 3) * 1000)
        p0 = 1000 - 2 * (p5 + p4 + p3 + p2 + p1)
        
        pos = [5]*p5 + [4]*p4 + [3]*p3 + [2]*p2 + [1]*p1
        neg = [-5]*p5 + [-4]*p4 + [-3]*p3 + [-2]*p2 + [-1]*p1
        e = pos + neg + [0]*p0
        lut[s, :] = e
        
    return lut

###########################

def get_lut_rpr(rpr_dict):
    lut = np.zeros(shape=(8, 8), dtype=np.int32)
    for x in range(8):
        for w in range(8):
            lut[x][w] = rpr_dict[(x, w)]
            
    return lut

###########################

def cconv(x, f, b, q, pool, stride, pad1, pad2, params):
    assert (stride == 1)

    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    y = np.zeros(shape=(Ho, Wo, Co))
    psum = 0
    
    ##################################################
    
    lut_rpr = get_lut_rpr(params['rpr'])
    lut_var = get_lut_var(params['sigma'], 32)
    print (lut_rpr)

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
    
    print (np.shape(patches), np.shape(f))
    
    ##################################################
    
    y, psum = pim(patches, f, (Ho * Wo, Co), lut_var, lut_rpr, params)
    y = np.reshape(y, (Ho, Wo, Co))
    
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = relu(y)
    y = avg_pool(y, pool)
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)

    return y, psum

##################################################

def cdot(x, w, b, q, params):
    H, W = np.shape(w)
    assert (len(x) == H)

    y = np.zeros(shape=(W, 1))
    psum = 0
    
    ##################################################
    
    lut_rpr = get_lut_rpr(params['rpr'])
    lut_var = get_lut_var(params['sigma'], 32)
    print (lut_rpr)

    ##################################################

    pb = []
    for xb in range(params['bpa']):
        pb.append(np.bitwise_and(np.right_shift(x.astype(int), xb), 1))
    
    x = np.stack(pb, axis=-1)
    nrow, nbit = np.shape(x)
    
    if (nrow % params['wl']):
        zeros = np.zeros(shape=(params['wl'] - (nrow % params['wl']), params['bpa']))
        x = np.concatenate((x, zeros), axis=1)
        
    x = np.reshape(x, (1, -1, params['wl'], params['bpa']))

    ##################################################
    
    w = w + params['offset']

    fb = []
    for wb in range(params['bpw']):
        fb.append(np.bitwise_and(np.right_shift(w.astype(int), wb), 1))
        
    w = np.stack(fb, axis=-1)
    
    nrow, ncol, nbit = np.shape(w)
    if (nrow % params['wl']):
        zeros = np.zeros(shape=(params['wl'] - (nrow % params['wl']), ncol, nbit))
        w = np.concatenate((w, zeros), axis=0)

    nrow, ncol, nbit = np.shape(w)
    w = np.reshape(w, (-1, params['wl'], ncol, nbit))

    nwl, wl, ncol, nbit = np.shape(w)
    w = np.transpose(w, (0, 1, 3, 2))
    w = np.reshape(w, (nwl, params['wl'], nbit * ncol))
    
    nwl, wl, ncol = np.shape(w)
    if (ncol % params['bl']):
        zeros = np.zeros(shape=(nwl, params['wl'], params['bl'] - (ncol % params['bl'])))
        w = np.concatenate((w, zeros), axis=2)

    w = np.reshape(w, (nwl, params['wl'], -1, params['bl']))
    
    ##################################################
    
    print (np.shape(x), np.shape(w))
    
    ##################################################
    
    y, psum = pim(x, w, (1, W), lut_var, lut_rpr, params)
    y = np.reshape(y, W)
        
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = relu(y)
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)

    return y, psum

def pim(x, w, y_shape, lut_var, lut_rpr, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    lut_var = np.ascontiguousarray(lut_var, np.int32)
    lut_rpr = np.ascontiguousarray(lut_rpr, np.int32)

    # print (lut_rpr)
    # print (np.array(lut_rpr.ctypes.strides))
    # assert (False)

    psum = lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(lut_var.ctypes.data), 
    ctypes.c_void_p(lut_rpr.ctypes.data), 
    ctypes.c_int(params['adc']),
    ctypes.c_int(params['skip']),
    ctypes.c_int(nrow),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))

    return y, psum
    
    
    
