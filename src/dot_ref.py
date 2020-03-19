
import numpy as np
from conv_utils import conv_output_length

from scipy.stats import norm, binom

##################################################

def relu(x):
    return x * (x > 0)
    
def avg_pool(x, p):
  H, W, C = np.shape(x)
  x = np.reshape(x, (H // p, p, W // p, p, C))
  x = np.transpose(x, (0, 2, 1, 3, 4))
  x = np.mean(x, axis=(2, 3))
  return x

def conv_ref(x, f, b, q, pool, stride, pad1, pad2):
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
            y[h, w, :] = relu(patch @ f_matrix)

    y = avg_pool(y, pool)
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)
    
    return y

def dot_ref(x, w, b, q):
    y = x @ w
    y = y / q
    y = np.floor(y)
    y = np.clip(y, -128, 127)
    return y
    
##################################################




