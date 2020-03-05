
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
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))
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

'''
def pcm2pcm(pcm, path):
    for ii in range(len(pcm)):
        assert(np.shape(pcm[ii]) == (8192, 32)) # 8192x32x4 -> 1024x1024
        
        pcm_ii = np.copy(pcm[ii])
        pcm_ii = pcm_ii + 8
        
        pcm_ii = np.reshape(pcm_ii, (8192, 32))
        pcm_ii_0 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 0), 1)
        pcm_ii_1 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 1), 1)
        pcm_ii_2 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 2), 1)
        pcm_ii_3 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 3), 1)
        pcm_ii = np.stack((pcm_ii_0, pcm_ii_1, pcm_ii_2, pcm_ii_3), axis=2)        
        assert(np.shape(pcm_ii) == (8192, 32, 4))
        
        pcm_ii = np.reshape(pcm_ii, (8, 1024, 128))
        pcm_ii = np.transpose(pcm_ii, (1, 2, 0))
        
        pcm_ii = np.reshape(pcm_ii, (1024, 1024))
        np.savetxt("%s/pcm%d.csv" % (path, ii+1), pcm_ii, fmt='%d', delimiter=" ")
'''

def pim_dot(x, w, params):
    # we need to do something like above.
    # reshape -> [N, 4]
    # multiply last dimension [8, 4, 2, 1]
    # offset

    y = 0
    psum = 0
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        for r1 in range(0, len(xb), 128):
            r2 = min(r1 + 128, len(xb))
            xbw = xb[r1:r2]
            pim, p = pim_kernel(xbw, w, params)
            assert (np.all(pim == (xbw @ w)))
            y += np.left_shift(pim.astype(int), b)
            psum += p
            
    return y, psum

##################################################

def pim_kernel(x, w, params):
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

        pdot = wl @ w
        psum += 1
        
        flag = (not flag) and (params['rpr'] > params['adc']) and (np.any(pdot == params['adc']))
        if not flag:
            wl_ptr = wl_stride[-1]
            y += pdot
        
    return y, psum

##################################################



