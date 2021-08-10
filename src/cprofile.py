
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes

profile_lib = ctypes.cdll.LoadLibrary('./profile.so')
profile_lib.profile.restype = ctypes.c_int

###########################
'''
def compress(adc):
    XB, WB, RPR, WL, ADC = np.shape(adc)
    ret = {}
    for xb in range(XB):
        ret[xb] = {}
        for wb in range(WB):
            ret[xb][wb] = {}
            for rpr in range(1, RPR): # +1 included here ... 64->65
                hist = adc[xb][wb][rpr][0:(rpr + 1), 0:(rpr + 1)]
                pmf = hist / np.sum(hist)
                ret[xb][wb][rpr] = pmf.astype(np.float16)
    return ret
'''
###########################
'''
def compress(adc):
    XB, WB, RPR, WL, ADC = np.shape(adc)
    ret = {}
    for xb in range(XB):
        ret[xb] = {}
        for wb in range(WB):
            hist = adc[xb][wb]
            pmf = np.copy(hist)
            pmf[1:] = pmf[1:] / np.sum(pmf[1:], axis=(1, 2), keepdims=True)
            ret[xb][wb] = pmf
    return ret
'''
###########################

def profile(layer, x, w, y_ref, y_shape, params):

    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape

    ################################

    y = np.zeros(shape=y_shape)
    adc = np.zeros(shape=(8, 8, params['max_rpr'] + 1, params['max_rpr'] + 1, params['max_rpr'] + 1))

    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)

    adc = np.ascontiguousarray(adc, np.int64)

    _ = profile_lib.profile(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(adc.ctypes.data),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(nrow),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))

    ################################

    rpr  = np.arange(1, params['max_rpr'] + 1)
    bits = np.sum(x, axis=2).reshape(nrow, nwl, xb, 1)
    row  = np.ceil(bits / rpr)
    row  = np.sum(row, axis=1)
    row  = np.mean(row, axis=0)

    ###########################

    ratio = np.count_nonzero(y_ref) / np.prod(np.shape(y_ref))

    ###########################
    
    profile = {'ratio': ratio, 'adc': adc, 'row': row}
    np.save('profile/%d' % (layer), profile)













