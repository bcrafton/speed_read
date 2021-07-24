
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

def cim(xb, wb, pb, params):

    # WB, 256 / WB = 32, 6=ECC, scale, C
    # 64 = 8 * 8

    N, NWL, WL, XB = np.shape(xb)
    NWL, WL, NBL, BL = np.shape(wb)
    _, _, _, BL_P = np.shape(pb)
    WB = 8
    C = NBL * BL // WB

    wb = np.reshape(wb, (NWL, WL, NBL * BL))
    pb = np.reshape(pb, (NWL, WL, NBL * BL_P))
    yb = np.zeros(shape=(N, C), dtype=np.int32)

    ################################################################

    WLs = np.sum(xb, axis=(2))
    rows = WLs / np.min(params['rpr'], axis=1)
    max_cycle = int(np.ceil(np.max(rows)))

    ################################################################

    count   = np.zeros(shape=(N, NWL, XB, WB, max_cycle), dtype=np.uint8)
    count   = np.ascontiguousarray(count.flatten(), np.uint8)

    error   = np.zeros(shape=(params['max_rpr']), dtype=np.uint32)
    error   = np.ascontiguousarray(error.flatten(), np.uint32)

    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    pb = np.ascontiguousarray(pb.flatten(), np.int8)
    yb = np.ascontiguousarray(yb.flatten(), np.int32)
    rpr = np.ascontiguousarray(params['rpr'].flatten(), np.uint8)
    var = np.ascontiguousarray(params['var'].flatten(), np.float32)
    conf = np.ascontiguousarray(params['conf'].flatten(), np.int32)
    value = np.ascontiguousarray(params['value'].flatten(), np.int32)

    ################################################################

    _ = cim_lib.cim(
    ctypes.c_void_p(xb.ctypes.data), 
    ctypes.c_void_p(wb.ctypes.data), 
    ctypes.c_void_p(pb.ctypes.data), 
    ctypes.c_void_p(yb.ctypes.data), 
    ctypes.c_void_p(count.ctypes.data), 
    ctypes.c_void_p(error.ctypes.data), 
    ctypes.c_void_p(rpr.ctypes.data), 
    ctypes.c_void_p(conf.ctypes.data), 
    ctypes.c_void_p(value.ctypes.data), 
    ctypes.c_int(max_cycle),
    ctypes.c_int(params['max_rpr']),
    ctypes.c_int(params['adc']),
    ctypes.c_int(N),
    ctypes.c_int(C),
    ctypes.c_int(NWL),
    ctypes.c_int(WL),
    ctypes.c_int(NBL),
    ctypes.c_int(BL),
    ctypes.c_int(BL_P))

    ################################################################

    yb      = np.reshape(yb,      (N, C))
    count   = np.reshape(count,   (N, NWL, XB, WB, max_cycle))
    error   = np.reshape(error,   (params['max_rpr']))
    print (error)

    ################################################################

    metrics = {}
    metrics['cycle'] = np.sum(count > 0)
    metrics['ron'] = 0
    metrics['roff'] = 0
    metrics['wl'] = np.sum(count)
    metrics['stall'] = 0
    metrics['block_cycle'] = np.sum(count > 0, axis=(0, 2, 3, 4))
    metrics['bb'] = (count > 0) * 1

    val, count = np.unique(count, return_counts=True)
    metrics['adc'] = np.zeros(shape=params['max_rpr']+1)
    for (v, c) in zip(val, count):
        metrics['adc'][v] = c

    return yb, metrics





















