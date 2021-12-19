
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

def cim(id, xb, wb, params):

    N, NWL, WL, XB = np.shape(xb)
    NWL, WL, NBL, BL = np.shape(wb)
    WB = 8
    C = NBL * BL // WB

    wb = np.reshape(wb, (NWL, WL, NBL * BL))
    yb = np.zeros(shape=(N, C), dtype=np.int32)

    ################################################################

    WLs = np.sum(xb, axis=(2))
    rows = WLs / np.min(params['rpr'], axis=1)
    max_cycle = int(np.ceil(np.max(rows)))

    ################################################################

    count = np.zeros(shape=(N, NWL, XB, WB, max_cycle), dtype=np.uint8)
    count = np.ascontiguousarray(count.flatten(), np.uint8)

    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    yb = np.ascontiguousarray(yb.flatten(), np.int32)

    assert (np.shape(params['rpr'])   == (XB, WB))
    assert (np.shape(params['conf'])  == (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))
    assert (np.shape(params['value']) == (XB, WB,                                               params['adc'] + 1))

    rpr = np.ascontiguousarray(params['rpr'].flatten(), np.uint8)
    conf = np.ascontiguousarray(params['conf'].flatten(), np.uint64)
    value = np.ascontiguousarray(params['value'].flatten(), np.float32)

    ################################################################

    _ = cim_lib.cim(
    ctypes.c_void_p(xb.ctypes.data), 
    ctypes.c_void_p(wb.ctypes.data), 
    ctypes.c_void_p(yb.ctypes.data), 
    ctypes.c_void_p(count.ctypes.data), 
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
    ctypes.c_int(BL))

    ################################################################

    yb    = np.reshape(yb,    (N, C))
    count = np.reshape(count, (N, NWL, XB, WB, max_cycle))
    conf = np.reshape(conf, (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))

    ################################################################

    metrics = {}
    metrics['cycle'] = np.sum(count > 0)
    # metrics['ron'] = 0
    # metrics['roff'] = 0
    # metrics['wl'] = np.sum(count)
    metrics['stall'] = 0
    # metrics['block_cycle'] = np.sum(count > 0, axis=(0, 2, 3, 4))
    # metrics['bb'] = (count > 0) * 1
    # metrics['count'] = count
    metrics['bb_cycles'] = np.sum(count > 0, axis=(2, 3, 4))
    metrics['vmm_cycles'] = np.sum(count > 0, axis=(0, 1, 4))

    vals, counts = np.unique(count, return_counts=True)
    metrics['adc'] = np.zeros(shape=params['max_rpr']+1)
    for (v, c) in zip(vals, counts):
        metrics['adc'][v] = c

    metrics['VMM_WL'] = np.zeros(shape=(XB, WB, params['max_rpr'] + 1))
    for xb in range(XB):
        for wb in range(WB):
            vals, counts = np.unique(count[:, :, xb, wb, :], return_counts=True)
            for (v, c) in zip(vals, counts):
                metrics['VMM_WL'][xb][wb][v] = c

    ##################################################################################

    flag = np.sum(metrics['VMM_WL'], axis=(0, 1)) == metrics['adc']
    assert (np.all(flag))

    flag = np.sum(metrics['VMM_WL'][:, :, 1:], axis=2) == metrics['vmm_cycles']
    assert (np.all(flag))

    ##################################################################################

    return yb, metrics





















