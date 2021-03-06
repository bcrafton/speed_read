
import numpy as np
import math
from conv_utils import *
from scipy.stats import norm

import ctypes
pim_lib = ctypes.cdll.LoadLibrary('./pim.cu.so')
pim_lib.pim.restype = ctypes.c_int

def pim_static(x, w, y_shape, lut_var, lut_rpr, alloc, lut_bias, params):

    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape

    ########

    x = x.astype(np.int8)
    w = np.reshape(w, (nwl, wl, ncol, 8)).astype(np.int8) # 8=Wb
    pim = np.zeros(shape=(nrow, ncol, 8, 8, nwl, params['max_rpr'] + 1), dtype=np.int8)

    ########

    # do we need to flatten or not ?
    # x = np.ascontiguousarray(x.flatten(), np.int8)
    x = np.ascontiguousarray(x, np.int8)
    w = np.ascontiguousarray(w, np.int8)
    pim = np.ascontiguousarray(pim, np.int8)

    ########

    ret = pim_lib.pim(
      ctypes.c_void_p(x.ctypes.data), 
      ctypes.c_void_p(w.ctypes.data), 
      ctypes.c_void_p(pim.ctypes.data), 
      ctypes.c_int(nrow),
      ctypes.c_int(ncol),
      ctypes.c_int(8), # Xb / Wb
      ctypes.c_int(nwl),
      ctypes.c_int(wl),
      ctypes.c_int(params['adc']),
      ctypes.c_int(params['max_rpr'] + 1)
    )

    #############################################################

    scale = 2 ** np.arange(0, 8)
    bit_weight = scale.reshape(-1, 1) * scale.reshape(1, -1)
    rpr_weight = np.arange(0, params['max_rpr'] + 1)
    offset = 128 * np.sum(scale * x, axis=(1, 2, 3))

    y = np.sum(pim * rpr_weight, axis=(4, 5))
    y = np.sum(y * bit_weight, axis=(2, 3))
    y = y - offset.reshape(-1, 1)

    #############################################################

    metrics_len = 13 + nwl
    metrics = np.zeros(shape=metrics_len)
    
    return y, metrics










