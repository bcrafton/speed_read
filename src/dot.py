
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
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :], p = dot(patch, f_matrix, b, q, params)
            psum += p
            
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

# wow okay lol
# how did we not run into this problem earlier.
# 4x4x3 < 128
# 2x2x32 = 128
# but now since we have more than 128 rows
# w is smaller than activations.
# so what do we do ? 
# > xbr = xb[r1:r2]
# > wr = w[r1:r2]

def pim_dot(x, w, params):
    y = 0
    psum = 0
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        for r1 in range(0, len(xb), 128):
            r2 = min(r1 + 128, len(xb))
            xbr = xb[r1:r2]
            wr = w[r1:r2]
            pim, p = pim_kernel(xbr, wr, params)
            # assert (np.all(pim == (xbr @ wr)))
            y += np.left_shift(pim.astype(int), b)
            psum += p
            
    return y, psum

##################################################

def pim_kernel(x, w, params):
    ishape, oshape, bpw = np.shape(w)
    assert(bpw == params['bpw'])
    w_matrix = np.reshape(w, (ishape, oshape * bpw))

    shift = 2 ** np.array(range(params['bpw']))

    wl_ptr = 0
    wl = np.zeros(len(x))
    wl_sum = np.zeros(len(x))
    wl_stride = np.zeros(len(x))
    
    y = 0
    psum = 0
    flag = False
    while wl_ptr < len(x):
        # advice = be careful about: (< vs <=), (> vs >=)
        wl[0] = x[0] & (wl_ptr <= 0)
        wl_sum[0] = x[0] & (wl_ptr <= 0)
        wl_stride[0] = (wl_sum[0] <= params['adc'])
        
        for ii in range(1, len(x)):
            if params['skip']:
                row = params['adc'] if flag else params['rpr']
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (wl_sum[ii - 1] < row)
                wl_sum[ii]    = (x[ii] & (wl_ptr <= ii)) + wl_sum[ii - 1]
                wl_stride[ii] = (wl_sum[ii] <= row) + wl_stride[ii - 1]
            else:
                assert (params['rpr'] == params['adc'])
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (ii < (wl_ptr + params['adc']))
                wl_stride[ii] = wl_ptr + params['adc']

        x_offset = np.sum(wl).astype(int) << (params['bpw'] - 1)

        pdot = wl @ w_matrix
        pdot_sum = pdot.reshape(oshape, params['bpw']) @ shift
        psum += 1
        
        flag = (not flag) and (params['rpr'] > params['adc']) and (np.any(pdot == params['adc']))
        if not flag:
            wl_ptr = wl_stride[-1]
            y += pdot_sum - x_offset
        
    return y, psum

##################################################



