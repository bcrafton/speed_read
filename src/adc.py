
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.stats import norm
import matplotlib.pyplot as plt
from kmeans import kmeans

####################################################
'''
def confusion(params):
    RPR = params['max_rpr']
    LRS = params['lrs']
    HRS = params['hrs']
    eps = 1e-12
    conf = np.zeros(shape=(RPR+1, RPR+1, RPR+1))
    for wl in range(RPR+1):
        for on in range(RPR+1):
            for adc in range(RPR+1):
                off = wl - on
                var = on*LRS**2 + off*HRS**2
                std = max(eps, np.sqrt(var))
                p = norm.cdf(adc + 0.5, on, std) - norm.cdf(adc - 0.5, on, std)
                conf[wl, on, adc] = p if (p > eps) else 0.

    scale = np.min(np.where(conf > 0, conf, np.inf), axis=-1, keepdims=True)
    assert (np.all(scale > 0))
    assert (np.all(np.isinf(scale) == False))
    assert (np.all(np.isnan(scale) == False))
    conf = (conf / scale).astype(int)
    assert (np.all(conf >= 0))
    return conf
'''
####################################################
'''
params = {
'lrs': 0.05, 
'hrs': 0.03,
'max_rpr': 8
}

conf = confusion(params)
print (conf)
np.savetxt('tmp', conf.reshape(81, 9), fmt='%d')
'''
####################################################

'''
def confusion(THRESH, RPR, ADC, HRS, LRS):
    eps = 1e-12
    conf = np.zeros(shape=(RPR, RPR, RPR, ADC))
    for rpr in range(RPR):
        for wl in range(RPR):
            for on in range(RPR):
                for adc in range(ADC):
                    off = wl - on
                    var = on*LRS**2 + off*HRS**2
                    std = max(eps, np.sqrt(var))
                    # is this indexing correct ? (rpr, adc + 1)
                    p = norm.cdf(THRESH[rpr, adc + 1], on, std) - norm.cdf(THRESH[rpr, adc], on, std)
                    conf[rpr, wl, on, adc] = p if (p > eps) else 0.

    scale = np.min(np.where(conf > 0, conf, np.inf), axis=-1, keepdims=True)
    assert (np.all(scale > 0))
    assert (np.all(np.isinf(scale) == False))
    assert (np.all(np.isnan(scale) == False))
    conf = (conf / scale).astype(int)
    assert (np.all(conf >= 0))
    return conf
'''

####################################################

def check(x):
    flag1 = np.any(np.isinf(x))
    flag2 = np.any(np.isnan(x))
    flag3 = not np.all(x >= 0.)
    if flag1 or flag2 or flag3: print (np.shape(x))
    assert (np.any(np.isinf(x)) == False)
    assert (np.any(np.isnan(x)) == False)
    assert (np.all(x >= 0.))

def confusion(THRESH, RPR, ADC, HRS, LRS):
    eps1 = 1e-12
    eps2 = 1e-6
    assert (len(THRESH) == ADC+2)

    wl  = np.arange(0, RPR+1).reshape(RPR+1,     1,     1).astype(int)
    on  = np.arange(0, RPR+1).reshape(    1, RPR+1,     1).astype(int)
    adc = np.arange(0, ADC+1).reshape(    1,     1, ADC+1).astype(int)
    off = np.maximum(0, wl - on)

    assert (np.all(THRESH[adc + 1] > THRESH[adc]))

    var = on*(LRS ** 2) + off*(HRS ** 2)
    check(var)

    std = np.maximum(eps1, np.sqrt(var))
    check(std)

    # should we assert ... sum(conf) == 1 ... ?
    conf = norm.cdf(THRESH[adc + 1], on, std) - norm.cdf(THRESH[adc], on, std)
    assert np.all(np.isclose(np.sum(conf, axis=-1), 1))
    check(conf)

    scale_max = np.max(                               conf, axis=-1, keepdims=True)
    scale_min = np.min(np.where(conf > eps2, conf, np.inf), axis=-1, keepdims=True)
    scale = np.minimum(scale_min, scale_max)
    check(scale)

    # will overflow here if > 2 ** 32
    # conf = (conf / scale).astype(np.uint32)
    conf = (conf / scale)
    assert (np.all(conf <= 0xFFFFFFFF))
    conf = conf.astype(np.uint32)
    check(conf)

    return conf

####################################################

def thresholds(counts, adc, method='normal'):
    on_counts = np.sum(counts, axis=0)
    if   method == 'normal': center = thresholds_normal(on_counts, adc)
    elif method == 'kmeans': center = thresholds_kmeans(on_counts, adc)
    else:                    assert (False)
    low  = center[:-1]
    high = center[1:]
    thresh = (high + low) / 2.
    thresh = [-np.inf] + thresh.tolist() + [np.inf]
    return np.array(thresh), np.array(center)

####################################################

def thresholds_kmeans(counts, adc):
    if (adc + 1) >= np.count_nonzero(counts): return np.arange(0, adc + 1)
    values = np.arange(0, len(counts))
    centroids = kmeans(values=values, counts=counts, n_clusters=adc + 1)
    centroids.sort()
    return centroids

# step is passed as (2 ** step)
# example -> [rpr=32, adc=16, step=2]
# A: [0,2,4,...32]
# B: [0,2,4,...16]
# which should it be ? 
# its currently (B).
# 
# update: we scrapped step
# 1: [0 .. adc]
# 2: [0 .. adc] * len(counts) / 2
# 3: sweep through all valid options and choose what minimizes error.
def thresholds_normal(counts, adc):
    centroids = np.arange(0, adc + 1)
    return centroids

####################################################


























