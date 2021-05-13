
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

def cim(xb, wb, rpr, var):

    N, NWL, WL, XB = np.shape(xb)
    NWL, WL, NBL, BL = np.shape(wb)

    wb = np.reshape(wb, (NWL, WL, NBL * BL))
    BL = NBL * BL

    cim_ref = np.zeros(shape=(N, NWL, XB, BL, 64), dtype=np.uint8)
    cim_var = np.zeros(shape=(N, NWL, XB, BL, 64), dtype=np.uint8)

    ################################################################

    cim_ref = np.ascontiguousarray(cim_ref.flatten(), np.uint8)
    cim_var = np.ascontiguousarray(cim_var.flatten(), np.uint8)

    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    rpr_table = np.ascontiguousarray(rpr.flatten(), np.uint8)
    var_table = np.ascontiguousarray(var.flatten(), np.float32)

    ################################################################

    _ = cim_lib.cim(
    ctypes.c_void_p(xb.ctypes.data), 
    ctypes.c_void_p(wb.ctypes.data), 
    ctypes.c_void_p(cim_ref.ctypes.data), 
    ctypes.c_void_p(cim_var.ctypes.data), 
    ctypes.c_void_p(rpr_table.ctypes.data), 
    ctypes.c_void_p(var_table.ctypes.data), 
    ctypes.c_int(N),
    ctypes.c_int(NWL),
    ctypes.c_int(WL),
    ctypes.c_int(BL))

    cim_ref = np.reshape(cim_ref, (N, NWL, XB, BL // 8, 8, 64))
    cim_var = np.reshape(cim_var, (N, NWL, XB, BL // 8, 8, 64))

    ################################################################

    '''
    total_error = np.count_nonzero(cim_ref - cim_var)
    total = np.count_nonzero(cim_ref)
    print (total_error / total)
    '''

    '''
    val, count = np.unique(np.count_nonzero(cim_ref - cim_var, axis=3), return_counts=True)
    print (val, count)
    '''

    '''
    # set (+1) in cim.c
    cim_var = np.maximum(cim_var.astype(int) - 1, 0)
    '''

    ################################################################

    scale = np.array([1,2,4,8,16,32,64,-128])
    scale = scale.reshape(-1, 1) * scale.reshape(1, -1)
    scale = np.reshape(scale, (8, 1, 8, 1))
    y = np.sum(cim_var * scale, axis=(1,2,4,5))

    return y





















