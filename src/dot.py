
import numpy as np
from conv_utils import conv_output_length

import ctypes
lib = ctypes.cdll.LoadLibrary('./cdot.so')
lib.pim_kernel.restype = ctypes.c_int
lib.conv.restype = ctypes.c_int
lib.pim.restype = ctypes.c_int

##################################################

def conv_ref(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))
    y = np.zeros(shape=(Ho, Wo, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            y[h, w, :] = dot_ref(patch, f_matrix, b, q)

    return y
    
def dot_ref(x, w, b, q):
    y = x @ w
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y

##################################################

def conv(x, f, b, q, stride, pad1, pad2, params):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    y = np.zeros(shape=(Ho, Wo, Co))
    psum = 0

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
    
    if (nrow % 256):
        zeros = np.zeros(shape=(npatch, 256 - (nrow % 256), nbit))
        patches = np.concatenate((patches, zeros), axis=1)
        
    patches = np.reshape(patches, (npatch, -1, 256, nbit))

    ##################################################
    
    f = f + 128

    f = np.reshape(f, (Fh * Fw * Ci, Co))
    fb = []
    for wb in range(params['bpw']):
        fb.append(np.bitwise_and(np.right_shift(f.astype(int), wb), 1))
        
    f = np.stack(fb, axis=-1)
    
    nrow, ncol, nbit = np.shape(f)
    if (nrow % 256):
        zeros = np.zeros(shape=(256 - (nrow % 256), ncol, nbit))
        f = np.concatenate((f, zeros), axis=0)

    nrow, ncol, nbit = np.shape(f)
    f = np.reshape(f, (-1, 256, ncol, nbit))

    nwl, wl, ncol, nbit = np.shape(f)
    f = np.transpose(f, (0, 1, 3, 2))
    f = np.reshape(f, (nwl, wl, nbit * ncol))
    
    nwl, wl, ncol = np.shape(f)
    if (ncol % 256):
        zeros = np.zeros(shape=(nwl, wl, 256 - (ncol % 256)))
        f = np.concatenate((f, zeros), axis=1)

    f = np.reshape(f, (nwl, wl, -1, 256))
    
    ##################################################
    
    # print (np.shape(patches), np.shape(f))
    
    ##################################################
    
    y, psum = pim(patches, f, (1024, 32), params)
    y = np.reshape(y, (32, 32, 32))
    
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)

    return y, psum
            
##################################################
'''
def pim(x, w, y_shape, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)
    psum = 0

    for row in range(nrow):
        for b in range(params['bpa']):
            for wl in range(nwl):
                for bl in range(nbl):
                    pim, _ = pim_kernel(x[row, wl, :, b], w[wl, :, bl, :], params)
                    y[row] += np.left_shift(pim.astype(int), b)

    return y, psum

def pim_kernel(x, w, params):
    shift = 2 ** np.array(range(params['bpw']))

    wl_ptr = 0
    y = 0
    psum = 0
    while wl_ptr < len(x):
        wl_sum = 0
        pdot = np.zeros(params['bl'])
        while (wl_ptr < len(x)) and (wl_sum + x[wl_ptr] <= params['adc']):
            if (x[wl_ptr]):
                wl_sum += 1
                pdot += w[wl_ptr]

            wl_ptr += 1

        psum += 1
        x_offset = wl_sum * params['offset']
        pdot_sum = pdot.reshape(8, 32).transpose(1, 0) @ shift
        y += pdot_sum - x_offset

    return y, psum
'''
##################################################

def pim(x, w, y_shape, params):
    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)

    psum = lib.pim(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_int(nrow),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))

    return y, psum

##################################################




















