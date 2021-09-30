
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ctypes
cim_lib = ctypes.cdll.LoadLibrary('./c.cim.so')
cim_lib.cim.restype = ctypes.c_int

################################################################

def cim(xb, wb, params):

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

    xb = np.ascontiguousarray(xb.flatten(), np.int8)
    wb = np.ascontiguousarray(wb.flatten(), np.int8)
    yb = np.ascontiguousarray(yb.flatten(), np.int32)

    assert (np.shape(params['rpr'])   == (XB, WB))
    assert (np.shape(params['conf'])  == (XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1, params['adc'] + 1))
    assert (np.shape(params['value']) == (XB, WB,                                               params['adc'] + 1))

    rpr = np.ascontiguousarray(params['rpr'].flatten(), np.uint8)
    conf = np.ascontiguousarray(params['conf'].flatten(), np.uint64)
    value = np.ascontiguousarray(params['value'].flatten(), np.float32)
    dist = np.zeros_like(conf)

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

    ################################################################
    '''
    print (np.sum(dist))
    print (np.sum(mean))
    '''
    ################################################################

    p_ref = np.sum(dist, axis=-1)

    profile = np.load('./profile/0.npy', allow_pickle=True).item()
    p = np.zeros(shape=(XB, WB, params['max_rpr'] + 1, params['max_rpr'] + 1))
    row = np.zeros(shape=(XB, WB))
    for i in range(XB):
        for j in range(WB):
            k = params['rpr'][i][j]
            k = np.where(k == params['rprs'])[0][0]
            p[i][j] = profile['adc'][i, j, k, :, :]
            row[i][j] = profile['row_avg'][i, params['rpr'][i][j] - 1]

    '''
    print ('P')
    # print (np.sum(p_ref))
    # print (np.sum(p))
    print (np.sum(p_ref - p))
    '''

    dump = {}
    dump['value'] = params['value']
    dump['conf'] = params['conf']
    dump['profile'] = p
    dump['q'] = params['q']
    dump['ratio'] = params['ratio']
    np.save('dump', dump)
    # assert (False)

    ################################################################

    on    = np.arange(params['max_rpr'] + 1).reshape((1, 1, 1, -1, 1))
    value = np.reshape(params['value'], (XB, WB, 1, 1, -1))
    e = (value - on)

    '''
    conf_sums = np.sum(conf, axis=-1, keepdims=True)
    dist_sums = np.sum(dist, axis=-1, keepdims=True)

    mask = (conf_sums > 0) * (dist_sums > 0)
    conf = np.where(mask, conf / (conf_sums + 1e-6), 0)
    dist = np.where(mask, dist / (dist_sums + 1e-6), 0)

    conf = conf / np.sum(conf) * e
    dist = dist / np.sum(dist) * e
    # print (np.sum(e > 0), np.sum(e < 0))
    # print (np.sum(np.abs(conf)))
    # print (np.sum(np.abs(dist)))
    print ('PE * E')
    print (np.around(np.sum(conf, axis=(2, 3, 4)), 3))
    print (np.around(np.sum(dist, axis=(2, 3, 4)), 3))
    '''

    ################################################################

    p = p / (np.sum(p, axis=(2, 3), keepdims=True) + 1e-6)
    p = p.reshape(8, 8, 65, 65, 1)

    scale = 2 ** np.arange(8)
    scale = scale.reshape(-1,1,1,1,1) * scale.reshape(1,-1,1,1,1)

    ratio = np.sum(yb > 0) / np.prod(np.shape(yb))

    pe = params['conf']
    pe = pe / np.sum(pe, axis=-1, keepdims=True)

    # print (row)
    row = row.reshape(8, 8, 1, 1, 1)

    mae_p = np.sum(p * pe * (e > 0) * e * scale * ratio / params['q'] * row)
    mae_n = np.sum(p * pe * (e < 0) * e * scale * ratio / params['q'] * row)
    mae = np.sum(p * pe * np.abs(e) * scale * ratio / params['q'] * row)
    # print ('mae', mae, mae_p, mae_n)

    ################################################################

    EP  = np.sum((p * pe > 0) * (e > 0) * np.abs(e), axis=(2, 3, 4))
    EN  = np.sum((p * pe > 0) * (e < 0) * np.abs(e), axis=(2, 3, 4))
    E = 2 * np.minimum(EP, EN) / np.maximum(1e-10, np.sum(EP + EN))
    # print (EP)
    # print (EN)

    PEP = np.sum(p * pe * (e > 0), axis=(2, 3, 4))
    PEN = np.sum(p * pe * (e < 0), axis=(2, 3, 4))
    PE = 2 * np.minimum(PEP, PEN)

    '''
    print ('E', np.sum(E))
    print ('PE', np.sum(PE))
    print (np.sum(E) * np.sum(PE))
    print (np.sum(E * PE))
    '''

    ################################################################

    # print ('PE')
    # print (np.around(error / mean, 3))
    # print (np.around(params['exp_p'], 3))
    # print (np.sum(error / mean))

    # PE = np.sum(error / mean * (1 - bias_mean) ** 2)
    # PE = np.sum(error / mean * np.maximum(0, 1 - 2 * bias_mean))
    # print (PE)

    ################################################################
    '''
    PE = np.sum(p * pe * (np.abs(e) > 0), axis=(2, 3, 4))
    print (np.around(PE, 3))
    print ()

    for WL in range(16):
        PE_XW    = np.sum(p[:, :, WL, :, :] * pe[:, :, WL, :, :] * np.abs(e[:, :, 0, :, :]), axis=(2, 3))
        PE_XW_MU = np.sum(p[:, :, WL, :, :] * pe[:, :, WL, :, :] *        e[:, :, 0, :, :] , axis=(2, 3))
        bias = np.abs(PE_XW_MU) / (PE_XW + 1e-6)
        print (np.around(bias, 3))
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





















