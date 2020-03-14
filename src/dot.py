
import numpy as np
from conv_utils import conv_output_length

import ctypes
lib = ctypes.cdll.LoadLibrary('./cdot.so')
lib.pim_kernel.restype = ctypes.c_int
lib.conv.restype = ctypes.c_int

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

def conv_ref2(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    assert (Hi == Wi)
    assert (Fh == Fw)
    X = Hi
    K = Fh
    
    X = X + pad1 + pad2
    Y = X - 2 * (K // 2)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f = np.reshape(f, (Fh * Fw * Ci, Co))
    y = np.zeros(shape=(Y, Y, Co))

    ############################

    x = np.ascontiguousarray(x, np.int32)
    f = np.ascontiguousarray(f, np.int32)
    y = np.ascontiguousarray(y, np.int32)
    
    # [2048  256    4]
    # which is what we want ...
    # print ('y strides', np.array(y.ctypes.strides))
    # print ('x strides', np.array(x.ctypes.strides))
    # print ('f strides', np.array(f.ctypes.strides))
    
    lib.conv(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(f.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data),
    ctypes.c_int(stride),
    ctypes.c_int(X),
    ctypes.c_int(Y),
    ctypes.c_int(K),
    ctypes.c_int(Ci),
    ctypes.c_int(Co))

    return y

##################################################
    
def conv(x, f, b, q, stride, pad1, pad2, params):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co, bpw = np.shape(f)
    assert (bpw == params['bpw'])
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co, params['bpw']))
    y = np.zeros(shape=(Ho, Wo, Co))
    psum = 0

    for h in range(Ho):        
        for w in range(Wo):
            # print ("(%d, %d)" % (h, w))
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :], p = dot(patch, f_matrix, b, q, params)
            psum += p
            # print (y[h, w, :])
            
    return y, psum

def dot(x, w, b, q, params):
    y, psum = pim_dot(x, w, params)
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y, psum
            
##################################################

def pim_dot(x, w, params):
    nrow, ncol, nbit = np.shape(w)
    y = np.zeros(shape=ncol)
    psum = 0
    
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        # print ("%d | %d/%d" % (b, np.sum(xb), np.shape(xb)[0]))

        for r1 in range(0, len(xb), params['wl']):
            r2 = min(r1 + params['wl'], len(xb))
            xbr = xb[r1:r2]
            wr = w[r1:r2]
        
            assert (params['wpb'] == (params['bl'] // params['bpw']))
            for c1 in range(0, ncol, params['wpb']):
                c2 = min(c1 + params['wpb'], ncol)
                wrc = wr[:, c1:c2]
        
                pim, p = pim_kernel(xbr, wrc, b, params)
                # assert (np.all(pim == (xbr @ wr)))
                y[c1:c2] += np.left_shift(pim.astype(int), b)
                psum += p
            
    return y, psum

##################################################

def pim_kernel(x, w, b, params):
    ishape, oshape, bpw = np.shape(w)
    assert(bpw == params['bpw'])
    w_matrix = np.reshape(w, (ishape, oshape * bpw))

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
                pdot += w_matrix[wl_ptr]

            wl_ptr += 1

        psum += 1
        x_offset = wl_sum * params['offset']
        pdot_sum = pdot.reshape(oshape, params['bpw']) @ shift
        y += pdot_sum - x_offset

    return y, psum

##################################################

'''
def pim_kernel(x, w, b, params):
    assert (np.shape(w) == (27, 32, 8))
    w = np.reshape(w, (27, 32 * 8))
    
    x = x.astype(np.int32)
    w = w.astype(np.int32)
    y = np.zeros(shape=32, dtype=np.int32)

    psum = lib.pim_kernel(
             ctypes.c_void_p(x.ctypes.data), 
             ctypes.c_void_p(w.ctypes.data), 
             ctypes.c_int(27),
             ctypes.c_int(256),
             ctypes.c_void_p(y.ctypes.data))

    return y, psum
'''























