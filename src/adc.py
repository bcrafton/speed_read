
import numpy as np
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

def confusion(THRESH, RPR, ADC, HRS, LRS):
    eps = 1e-12

    rpr = np.arange(0, RPR).reshape(RPR,   1,   1,   1)
    wl  = np.arange(0, RPR).reshape(  1, RPR,   1,   1)
    on  = np.arange(0, RPR).reshape(  1,   1, RPR,   1)
    adc = np.arange(0, ADC).reshape(  1,   1,   1, ADC)

    off = wl - on
    var = on * (LRS ** 2) + off * (HRS ** 2)
    std = np.maximum(eps, np.sqrt(var))

    conf = norm.cdf(THRESH[rpr, adc + 1], on, std) - norm.cdf(THRESH[rpr, adc], on, std)
    assert (np.all(conf >= 0.))
    assert (np.all(np.isinf(conf) == False))
    assert (np.all(np.isnan(conf) == False))

    scale = np.min(np.where(conf > eps, conf, np.inf), axis=-1, keepdims=True)
    assert (np.all(scale > 0.))
    assert (np.all(np.isinf(scale) == False))
    assert (np.all(np.isnan(scale) == False))

    conf = (conf / scale).astype(int)
    assert (np.all(conf >= 0.))
    assert (np.all(np.isinf(conf) == False))
    assert (np.all(np.isnan(conf) == False))

    return conf

####################################################

def thresholds(counts, adc, method='normal'):
    RPR, WL, ON = np.shape(counts)
    thresh = np.zeros(shape=(RPR, adc + 1))
    value = np.zeros(shape=(RPR, adc))
    for rpr in range(0, RPR):
        thresh[rpr], value[rpr] = thresholds_help(np.sum(counts[rpr], axis=0), adc, method)
    return thresh, value

####################################################

def thresholds_help(counts, adc, method='normal'):
    if   method == 'normal': center = thresholds_normal(counts, adc)
    elif method == 'step':   center = thresholds_step(counts, adc)
    elif method == 'kmeans': center = thresholds_kmeans(counts, adc)
    else:                    assert (False)
    low  = center[:-1]
    high = center[1:]
    thresh = (high + low) / 2.
    thresh = [-np.inf] + thresh.tolist() + [np.inf]
    return thresh, center

####################################################

def thresholds_kmeans(counts, adc):
    values = np.arange(0, len(counts))
    if adc >= len(counts):              return np.arange(0, adc)
    if adc >= np.count_nonzero(counts): return np.arange(0, adc)
    centroids = kmeans(values=values, counts=counts, n_clusters=adc)
    centroids.sort()
    return centroids

def thresholds_step(counts, adc):
    assert (False)
    return centroids

def thresholds_normal(counts, adc):
    centroids = np.arange(0, len(adc))
    return centroids

####################################################

























