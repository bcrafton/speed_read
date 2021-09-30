
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

    error = np.zeros(shape=(XB, WB), dtype=np.uint64) 
    error = np.ascontiguousarray(error.flatten(), np.uint64)

    mean = np.zeros(shape=(XB, WB), dtype=np.uint64) 
    mean = np.ascontiguousarray(mean.flatten(), np.uint64)

    dist = np.zeros_like(params['conf'])

    dot = np.zeros(shape=(N, NWL, XB, NBL, BL, max_cycle, 3), dtype=np.uint8)

    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    yb = np.ascontiguousarray(yb.flatten(), np.int32)

    assert (np.shape(params['rpr'])   == (XB, WB))
    assert (np.shape(params['conf'])  == (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))
    assert (np.shape(params['value']) == (XB, WB,                                               params['adc'] + 1))

    rpr = np.ascontiguousarray(params['rpr'].flatten(), np.uint8)
    conf = np.ascontiguousarray(params['conf'].flatten(), np.uint64)
    value = np.ascontiguousarray(params['value'].flatten(), np.float32)
    dist = np.ascontiguousarray(dist.flatten(), np.uint64)
    dot = np.ascontiguousarray(dot.flatten(), np.uint8)

    ################################################################

    _ = cim_lib.cim(
    ctypes.c_void_p(xb.ctypes.data), 
    ctypes.c_void_p(wb.ctypes.data), 
    ctypes.c_void_p(yb.ctypes.data), 
    ctypes.c_void_p(count.ctypes.data), 
    ctypes.c_void_p(error.ctypes.data),
    ctypes.c_void_p(mean.ctypes.data), 
    ctypes.c_void_p(rpr.ctypes.data), 
    ctypes.c_void_p(conf.ctypes.data), 
    ctypes.c_void_p(value.ctypes.data), 
    ctypes.c_void_p(dist.ctypes.data), 
    ctypes.c_void_p(dot.ctypes.data), 
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
    error = np.reshape(error, (XB, WB))
    mean  = np.reshape(mean, (XB, WB))

    conf = np.reshape(conf, (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))
    dist = np.reshape(dist, (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))
    dot  = np.reshape(dot, (N, NWL, XB, NBL, BL, max_cycle, 3))
    
    ################################################################
    '''
    print (np.shape(dot))
    dot = np.reshape(dot, (N, NWL, XB, NBL*BL//WB, WB, max_cycle, 3))

    value = np.reshape(params['value'], (XB, WB, params['adc'] + 1))

    scale = np.array([1, 2, 4, 8, 16, 32, 64, -128])
    scale = scale.reshape(8, 1) * scale.reshape(1, 8)

    y_dot = np.zeros_like(yb)
    y_ref = np.zeros_like(yb)
    y_err = np.zeros_like(yb)
    for r in range(1024):
        print (r)
        for nwl in range(NWL):
            for xb in range(XB):
                for c in range(NBL*BL//WB):
                    for wb in range(WB):
                        for n in range(max_cycle):
                            wl  = dot[r][nwl][xb][c][wb][n][0]
                            on  = dot[r][nwl][xb][c][wb][n][1]
                            out = dot[r][nwl][xb][c][wb][n][2]
                            y_err[r][c] += scale[xb][wb] * (value[xb, wb, out] - on)
                            y_dot[r][c] += scale[xb][wb] * value[xb, wb, out]
                            y_ref[r][c] += scale[xb][wb] * on

    mean = np.mean(y_dot - y_ref) / params['q']
    mse = np.sqrt(np.mean( (y_dot - y_ref)**2 )) / params['q']
    print (mean, mse)
    '''
    ################################################################
    '''
    # use dist for p AND pe
    p_ref = np.sum(dist, axis=-1)

    profile = np.load('./profile/%d.npy' % (id), allow_pickle=True).item()
    p = np.zeros(shape=(XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1))
    row = np.zeros(shape=(XB, WB))
    for i in range(XB):
        for j in range(WB):
            k = params['rpr'][i][j]
            k = np.where(k == params['rprs'])[0][0]
            p[i][j] = profile['adc'][i, j, k, :, :]
            row[i][j] = profile['row_avg'][i, params['rpr'][i][j] - 1]

    print (np.ceil(row))
    assert (np.sum(dist) == np.sum(mean))
    assert (np.all(p_ref == p))
    '''
    ################################################################
    '''
    conf_sums = np.sum(conf, axis=-1, keepdims=True)
    dist_sums = np.sum(dist, axis=-1, keepdims=True)

    mask = (conf_sums > 0) * (dist_sums > 0)
    conf = np.where(mask, conf / (conf_sums + 1e-6), 0)
    dist = np.where(mask, dist / (dist_sums + 1e-6), 0)
    '''
    ################################################################
    '''
    on    = np.arange(params['max_rpr'] + 1).reshape((1, 1, 1, -1, 1))
    value = np.reshape(params['value'], (XB, WB, 1, 1, -1))
    e = (value - on)

    p = p / (np.sum(p, axis=(2, 3), keepdims=True) + 1e-6)
    p = p.reshape(8, 8, 65, 65, 1)

    p2 = p_ref
    p2 = p2 / (np.sum(p2, axis=(2, 3), keepdims=True) + 1e-6)
    p2 = p2.reshape(8, 8, 65, 65, 1)

    scale = 2 ** np.arange(8)
    scale = scale.reshape(-1,1,1,1,1) * scale.reshape(1,-1,1,1,1)

    row = row.reshape(8, 8, 1, 1, 1)

    pe1 = params['conf']
    pe1 = pe1 / (np.sum(pe1, axis=-1, keepdims=True) + 1e-6)

    pe2 = dist
    pe2 = pe2 / (np.sum(pe2, axis=-1, keepdims=True) + 1e-6)
    '''
    ################################################################
    '''
    error = e * scale / params['q']

    mean1 = np.sum(p * pe1 * error)
    mae1_xw = np.sum(p * pe1 * np.abs(error), axis=(2, 3, 4), keepdims=True)
    mean1_xw = np.sum(p * pe1 * error, axis=(2, 3, 4), keepdims=True)
    mse1 = np.sqrt(np.sum(p * pe1 * (error - mean1_xw) ** 2))
    print (mse1, mean1)

    mean2 = np.sum(p * pe2 * error)
    mae2_xw = np.sum(p * pe2 * np.abs(error), axis=(2, 3, 4), keepdims=True)
    mean2_xw = np.sum(p * pe2 * error, axis=(2, 3, 4), keepdims=True)
    mse2 = np.sqrt(np.sum(p * pe2 * (error - mean2_xw) ** 2))
    print (mse2, mean2)
    '''
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





















