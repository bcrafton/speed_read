
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

    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)

    y = np.zeros(shape=y_shape)
    y = np.ascontiguousarray(y, np.int32)

    rprs = np.ascontiguousarray(params['rprs'], np.int32)
    Nrpr = len(params['rprs'])

    adc = np.zeros(shape=(8, 8, Nrpr, params['max_rpr'] + 1, params['max_rpr'] + 1))
    adc = np.ascontiguousarray(adc, np.int64)

    _ = profile_lib.profile(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(adc.ctypes.data),
    ctypes.c_int(rprs.ctypes.data),
    ctypes.c_int(Nrpr),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(nrow),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))

    ################################

    rpr  = np.arange(1, params['max_rpr'] + 1)
    # sum over WL
    bits = np.sum(x, axis=2).reshape(nrow, nwl, xb, 1)
    # ceil it -> each N_WL is atleast 1 cycle
    row  = np.ceil(bits / rpr).astype(int)
    # sum over N_WL
    row = np.sum(row, axis=1)
    # avg over patches
    row_avg = np.mean(row, axis=0)

    ###########################

    rpr  = np.arange(1, params['max_rpr'] + 1)
    # sum over WL
    bits = np.sum(x, axis=2).reshape(nrow, nwl, xb, 1)
    # ceil it -> each N_WL is atleast 1 cycle
    row  = np.ceil(bits / rpr + 1e-6).astype(int)
    # sum over N_WL
    row = np.sum(row, axis=1)
    # create row pmf
    pmf = np.zeros(shape=(xb, params['max_rpr'], nwl * params['wl'])) # params['wl'] is actually low-ball ... could be (NWL * WL)
    for b in range(xb):
        for r in range(params['max_rpr']):
            val, count = np.unique(row[:, b, r], return_counts=True)
            for (v, c) in zip(val, count):
                pmf[b, r, v] = c
    # normalize
    pmf = pmf / np.sum(pmf, axis=2, keepdims=True)

    ###########################

    ratio = np.count_nonzero(y_ref) / np.prod(np.shape(y_ref))

    ###########################

    assert (wl == params['wl'])
    assert (bl == params['bl'])
    profile = {'ratio': ratio, 'adc': adc, 'row': pmf, 'row_avg': row_avg, 'max_rpr': params['max_rpr'], 'wl': params['wl'], 'bl': params['bl']}
    np.save('profile/%d' % (layer), profile)













