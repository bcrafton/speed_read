
import numpy as np
from conv_utils import *
from scipy.stats import norm

import ctypes

profile_lib = ctypes.cdll.LoadLibrary('./profile.so')
profile_lib.profile.restype = ctypes.c_int

###########################

def profile(x, w, y_shape, rpr_low, rpr_high, params):

    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape
        
    y = np.zeros(shape=y_shape)
    count_adc = np.zeros(shape=(8, 8, rpr_high+1, rpr_high+1))
    count_row = np.zeros(shape=(8, rpr_high+1, rpr_high+1))
    
    x = np.ascontiguousarray(x, np.int32)
    w = np.ascontiguousarray(w, np.int32)
    y = np.ascontiguousarray(y, np.int32)

    count_adc = np.ascontiguousarray(count_adc, np.int64)
    count_row = np.ascontiguousarray(count_row, np.int64)

    ########

    _ = profile_lib.profile(
    ctypes.c_void_p(x.ctypes.data), 
    ctypes.c_void_p(w.ctypes.data), 
    ctypes.c_void_p(y.ctypes.data), 
    ctypes.c_void_p(count_adc.ctypes.data),
    ctypes.c_void_p(count_row.ctypes.data),
    ctypes.c_int(rpr_high),
    ctypes.c_int(nrow),
    ctypes.c_int(ncol),
    ctypes.c_int(nwl),
    ctypes.c_int(nbl),
    ctypes.c_int(wl),
    ctypes.c_int(bl))

    return y, count_adc, count_row
    
    
    
