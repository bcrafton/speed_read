
import numpy as np
from conv_utils import *

from scipy.stats import norm, binom

##################################################

def conv_ref(x, f, b, q, pool, stride, pad1, pad2, relu_flag):
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
            y[h, w, :] = patch @ f_matrix

    '''
    if relu_flag:
        y = relu(y)
    y = avg_pool(y, pool)
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)
    '''
    
    return y

def dot_ref(x, w, b, q):
    y = x @ w
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)
    return y
    
##################################################




