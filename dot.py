
import numpy as np
from conv_utils import conv_output_length

##################################################

def conv_ref(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    y = np.zeros(shape=(Ho, Wo, Co))
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            y[h, w, :] = dot_ref(patch, f_matrix, b, q)

    return y

def dot_ref(x, w, b, q):
    y = x @ w
    assert(np.all(np.absolute(y) < 2 ** 15))
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

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    y = np.zeros(shape=(Ho, Wo, Co))
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :] = dot(patch, f_matrix, b, q, params)

    return y

def dot(x, w, b, q, params):
    y = pim_dot(x, w, params)
    assert(np.all(np.absolute(y) < 2 ** 15))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y
            
##################################################

def pim_dot(x, w, params):
    y = 0
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        for r1 in range(0, len(xb), 128):
            r2 = min(r1 + 128, len(xb))
            xbw = xb[r1:r2]
            pim = pim_kernel(xbw, w, params)
            assert (np.all(pim == (xbw @ w)))
            y += np.left_shift(pim.astype(int), b)
        
    return y

##################################################

def pim_kernel(x, w, params):
    wl_ptr = 0
    wl = np.zeros(len(x))
    wl_sum = np.zeros(len(x))
    wl_stride = np.zeros(len(x))
    
    y = 0
    while wl_ptr < len(x):
        # advice = be careful about: (< vs <=), (> vs >=)
        wl[0] = x[0] & (wl_ptr <= 0)
        wl_sum[0] = x[0] & (wl_ptr <= 0)
        wl_stride[0] = (wl_sum[0] <= 8)
        
        for ii in range(1, len(x)):
            if params['skip']:
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (wl_sum[ii - 1] < params['rpr'])
                wl_sum[ii]    = (x[ii] & (wl_ptr <= ii)) + wl_sum[ii - 1]
                wl_stride[ii] = (wl_sum[ii] <= params['rpr']) + wl_stride[ii - 1]
            else:
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (ii < (wl_ptr + params['rpr']))
                wl_stride[ii] = wl_ptr + params['rpr']

        wl_ptr = wl_stride[-1]
        y += wl @ w
        
    return y

##################################################



