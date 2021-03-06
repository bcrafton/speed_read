
import numpy as np
import math
from conv_utils import *
from scipy.stats import norm

import ctypes
profile_lib = ctypes.cdll.LoadLibrary('./profile.cu.so')
profile_lib.cu_profile.restype = ctypes.c_int

###########################

def profile(x, w, y_shape, rpr_low, rpr_high, params):

    nrow, nwl, wl, xb = np.shape(x)
    nwl, wl, nbl, bl = np.shape(w) # nwl, nbl, wl, bl
    nrow, ncol = y_shape

    ########

    w = np.reshape(w, (nwl, wl, nbl * bl // 8, 8)) # 8=Wb

    x = np.ascontiguousarray(x.flatten(), np.uint8)
    w = np.ascontiguousarray(w.flatten(), np.uint8)

    ########

    counts = []
    y0 = np.zeros(shape=(8, 8, rpr_high + 1), dtype=np.int64)
    y0 = np.ascontiguousarray(y0, np.int64)
    counts.append(y0)
    for rpr in range(rpr_low, rpr_high + 1):
        y = np.zeros(shape=(8, 8, rpr_high + 1), dtype=np.int64)
        y = np.ascontiguousarray(y, np.int64)
        ret = profile_lib.cu_profile(
          ctypes.c_void_p(x.ctypes.data), 
          ctypes.c_void_p(w.ctypes.data), 
          ctypes.c_void_p(y.ctypes.data), 
          ctypes.c_int(nrow),
          ctypes.c_int(ncol),
          ctypes.c_int(8), # Xb / Wb
          ctypes.c_int(nwl),
          ctypes.c_int(wl),
          ctypes.c_int(rpr),
          ctypes.c_int(rpr_high + 1)
        )
        assert (np.all(np.sum(y, axis=-1) > 0))
        counts.append(y)

    return np.stack(counts, axis=2)
