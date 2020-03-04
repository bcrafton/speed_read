
import numpy as np
np.set_printoptions(threshold=np.inf)

from conv_utils import conv_output_length
from dot import *
from defines import *

###########################################

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

###########################################

def sim(path):

    params = np.load('./%s/params.npy' % (path), allow_pickle=True).item()
    
    yout  = [[None for col in range(params['num_layer'])] for row in range(params['num_example'])] 

    for ii in range(params['num_example']):
        for jj in range(params['num_layer']):
            
            param = params[jj]
        
            #######
            
            xin = params['x'][ii] if (jj == 0) else yout[ii][jj-1]
            xin = np.reshape(xin, param['x'])
            xin = xin.astype(int)
            
            #######

            if param['op'] == OPCODE_CONV: 
                y     = conv(np.copy(xin), param['weights'], param['bias'], param['quant'], param['dims']['stride'], param['dims']['pad1'], param['dims']['pad2'])
                y_ref = conv_ref(np.copy(xin), param['weights'], param['bias'], param['quant'], param['dims']['stride'], param['dims']['pad1'], param['dims']['pad2'])
                assert (np.all(y == y_ref))
            else:
                y     = dot(np.copy(xin), param['weights'], param['bias'], param['quant'])
                y_ref = dot_ref(np.copy(xin), param['weights'], param['bias'], param['quant'])
                assert (np.all(y == y_ref))
                
            yout[ii][jj] = y
                
######################



















