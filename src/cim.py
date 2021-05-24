
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

def ecc_encode(x, d, p):
    assert (2 ** p >= d + p + 1)
    #######################################
    bit = []
    for i in range(1, d + 1):
        b = i
        for j in range(d):
            b += b >= (2 ** j)
        bit.append(b)
    bit = np.array(bit)
    #######################################
    ps = [None] * p
    for i in range(p):    
        sel = 1 * (np.bitwise_and(bit, 2**i) == 2**i)
        ps[i] = np.sum(sel * x, axis=-1) % 2
    ps = np.stack(ps, axis=-1)
    #######################################
    return ps 

################################################################

def ecc(data, data_ref, parity, parity_ref):
    #########################
    d = np.shape(data)[-1]
    p = np.shape(parity)[-1]
    
    bit = []
    for i in range(1, d + 1):
        b = i
        for j in range(d):
            b += b >= (2 ** j)
        bit.append(b)
    bit = np.array(bit)
    #########################
    cs = []
    for i in range(p):
        sel = 1 * (np.bitwise_and(bit, 2**i) == 2**i)
        c = (np.sum(sel * data, axis=-1) + parity[..., i]) % 2
        cs.append(c)
    cs = np.stack(cs, axis=-1)
    #########################
    scale = 2 ** np.arange(0, p)
    addr = np.sum(cs * scale, axis=-1, keepdims=True)
    #########################
    d_addr = (addr - bit) == 0
    p_addr = (addr - scale) == 0
    #########################
    data   = np.where(d_addr, data_ref,   data)
    parity = np.where(p_addr, parity_ref, parity)
    #########################
    return data, parity

################################################################

def cim(xb, wb, rpr, var):

    N, NWL, WL, XB = np.shape(xb)
    NWL, WL, NBL, BL = np.shape(wb)

    # wb = np.reshape(wb, (NWL, WL, NBL * BL))
    # BL = NBL * BL

    wb = np.reshape(wb, (NWL, WL, NBL * BL // 32, 32))
    pb = ecc_encode(wb, 32, 6)

    # print (np.shape(wb))
    # print (np.shape(pb))
    
    wb = np.reshape(wb, (NWL, WL, NBL * BL))
    pb = np.reshape(pb, (NWL, WL, NBL * BL // 32 * 6))
    _, _, BL_W = np.shape(wb)
    _, _, BL_P = np.shape(pb)

    wb = np.concatenate((wb, pb), axis=2)
    _, _, BL = np.shape(wb)

    # print (BL, BL_W, BL_P)

    ################################################################

    WLs = np.sum(xb, axis=(2))
    rows = WLs / np.min(rpr, axis=1)
    max_cycle = int(np.ceil(np.max(rows)))

    ################################################################

    cim_ref = np.zeros(shape=(N, NWL, XB, BL, max_cycle), dtype=np.uint8)
    cim_var = np.zeros(shape=(N, NWL, XB, BL, max_cycle), dtype=np.uint8)
    count   = np.zeros(shape=(N, NWL, XB, max_cycle), dtype=np.uint8)

    ################################################################

    cim_ref = np.ascontiguousarray(cim_ref.flatten(), np.uint8)
    cim_var = np.ascontiguousarray(cim_var.flatten(), np.uint8)
    count   = np.ascontiguousarray(count.flatten(), np.uint8)

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
    ctypes.c_void_p(count.ctypes.data), 
    ctypes.c_void_p(rpr_table.ctypes.data), 
    ctypes.c_void_p(var_table.ctypes.data), 
    ctypes.c_int(max_cycle),
    ctypes.c_int(N),
    ctypes.c_int(NWL),
    ctypes.c_int(WL),
    ctypes.c_int(BL))

    ################################################################

    cim_ref = np.reshape(cim_ref, (N, NWL, XB, BL, max_cycle))
    cim_var = np.reshape(cim_var, (N, NWL, XB, BL, max_cycle))
    count   = np.reshape(count,   (N, NWL, XB, max_cycle))

    # BB(count)

    ################################################################

    ecc_var = np.reshape(cim_var[:, :, :, BL_W:BL, :], (N, NWL, XB, BL_P //  6,  6, max_cycle)).transpose(0,1,2,3,5,4)
    cim_var = np.reshape(cim_var[:, :, :,  0:BL_W, :], (N, NWL, XB, BL_W // 32, 32, max_cycle)).transpose(0,1,2,3,5,4)

    ecc_ref = np.reshape(cim_ref[:, :, :, BL_W:BL, :], (N, NWL, XB, BL_P //  6,  6, max_cycle)).transpose(0,1,2,3,5,4)
    cim_ref = np.reshape(cim_ref[:, :, :,  0:BL_W, :], (N, NWL, XB, BL_W // 32, 32, max_cycle)).transpose(0,1,2,3,5,4)

    cim_var, ecc_var = ecc(cim_var, cim_ref, ecc_var, ecc_ref)

    cim_var = cim_var.transpose(0,1,2,3,5,4).reshape(N, NWL, XB, BL_W // 8, 8, max_cycle)

    ################################################################

    scale = np.array([1,2,4,8,16,32,64,-128])
    scale = scale.reshape(-1, 1) * scale.reshape(1, -1)
    scale = np.reshape(scale, (8, 1, 8, 1))
    y = np.sum(cim_var * scale, axis=(1,2,4,5))

    ################################################################

    metrics = {}
    metrics['cycle'] = np.sum(count > 0)
    metrics['ron'] = 0
    metrics['roff'] = 0
    metrics['wl'] = np.sum(count)
    metrics['stall'] = 0
    metrics['block_cycle'] = np.sum(count, axis=(0, 2, 3))
    metrics['bb'] = count

    val, count = np.unique(count, return_counts=True)
    metrics['adc'] = np.zeros(shape=8+1)
    for (v, c) in zip(val, count):
        metrics['adc'][v] = c

    return y, metrics





















