
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from kmeans import kmeans

####################################################

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
def confusion(THRESH, RPR, HRS, LRS):
    eps = 1e-12
    conf = np.zeros(shape=(RPR+1, RPR+1, RPR+1))
    for wl in range(RPR+1):
        for on in range(RPR+1):
            for thresh in THRESH:
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

def thresholds(counts, adc, method='normal'):
    if   method == 'normal': center = thresholds_normal(counts, adc)
    elif method == 'step':   center = thresholds_step(counts, adc)
    elif method == 'kmeans': center = thresholds_kmeans(counts, adc)
    else:                    assert (False)
    low  = center[:-1]
    high = center[1:]
    # print (center)
    thresh = (high + low) / 2.
    thresh = [-np.inf] + thresh.tolist() + [np.inf]
    # print (thresh)
    # print ()
    return thresh, center

####################################################

def thresholds_kmeans(counts, adc):
    values = np.arange(0, len(counts))
    if adc > len(counts):              return values
    if adc > np.count_nonzero(counts): return values
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


























