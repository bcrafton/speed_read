
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.stats import norm
import matplotlib.pyplot as plt
from kmeans import kmeans
import itertools
import math

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
    eps1 = 1e-20
    eps2 = 1e-10
    assert (len(THRESH) == ADC+1)

    wl  = np.arange(0, RPR+1).reshape(RPR+1,     1,   1).astype(int)
    on  = np.arange(0, RPR+1).reshape(    1, RPR+1,   1).astype(int)
    adc = np.arange(0,   ADC).reshape(    1,     1, ADC).astype(int)
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
    conf = conf * 100.
    assert (np.all(conf <= 0xFFFFFFFFFFFFFFFF))
    conf = conf.astype(np.uint64)
    check(conf)

    return conf

####################################################

def thresholds(counts, adc, sar, method='normal'):
    on_counts = np.sum(counts, axis=0)
    if   method == 'normal': center = thresholds_normal(on_counts, adc, sar)
    elif method == 'kmeans': center = thresholds_kmeans(on_counts, adc, sar)
    elif method == 'soft':   center = thresholds_kmeans_soft(on_counts, adc, sar)
    else:                    assert (False)
    low  = center[:-1]
    high = center[1:]
    thresh = (high + low) / 2.
    thresh = [-np.inf] + thresh.tolist() + [np.inf]
    return np.array(thresh), np.array(center)

####################################################

def thresholds_kmeans(counts, adc, sar):
    states = (adc + 1) ** sar
    if states >= np.count_nonzero(counts): return np.arange(0, states)
    values = np.arange(0, len(counts))
    centroids = kmeans(values=values, counts=counts, n_clusters=states)
    centroids.sort()
    return centroids

####################################################

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
def thresholds_normal(counts, adc, sar):
    states = (adc + 1) ** sar
    centroids = np.arange(0, states)
    return centroids

####################################################
'''
# need to include sar in here ?
# so k-means dosnt respect sar constraint ?
# it dosnt look like it ...
# if we ignore this constraint, can we make it work ? 
def thresholds_kmeans_soft(counts, adc, sar):
    max_rpr = len(counts) - 1
    max_sar = np.log2(max_rpr)
    assert (max_sar % 1 == 0)
    max_sar = int(max_sar)
    ###################################
    refs = []
    for ref in itertools.combinations(2 ** np.arange(max_sar), adc + sar - 1):
        configs = [0]
        for size in range(1, adc + sar):
            for config in itertools.combinations(ref, size):
                configs.append(np.sum(config))
        configs = np.array(configs)
        refs.append(configs)
    ###################################
    def compute_error(counts, ref):
        val = np.arange(0, len(counts))
        pmf = counts / np.sum(counts)
        diff = val - ref.reshape(-1, 1)
        diff = np.abs(diff)
        diff = np.min(diff, axis=0)
        error = np.sum(pmf * diff)
        return error
    ###################################
    best_ref = None; best_error = np.inf
    for ref in refs:
        error = compute_error(counts, ref)
        if error < best_error:
            best_error = error
            best_ref = ref
    ###################################
    assert best_ref is not None
    best_ref.sort()
    return best_ref
'''
####################################################

def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)

# need to include sar in here ?
# so k-means dosnt respect sar constraint ?
# it dosnt look like it ...
# if we ignore this constraint, can we make it work ? 
def thresholds_kmeans_soft(counts, adc, sar):
    max_rpr = len(counts) - 1
    max_sar = math.log(max_rpr, adc + 1)
    max_sar = int(np.ceil(max_sar))
    ###################################
    choices = []
    for s in range(max_sar):
        choice = [a * (adc+1) ** s for a in range(adc + 1)]
        choices.append(choice)
    ###################################
    refs = []
    for comb in itertools.combinations(choices, sar):
        comb_refs = []
        for comb_ref in comb:
            if comb_refs: comb_refs = list(itertools.product(comb_ref, comb_refs))
            else:         comb_refs = comb_ref
        ref = []
        for comb_ref in comb_refs:
            ref.append( np.sum(flatten(comb_ref)) )
        ref = np.array(ref)
        refs.append(ref)
    ###################################
    def compute_error(counts, ref):
        val = np.arange(0, len(counts))
        pmf = counts / np.sum(counts)
        diff = val - ref.reshape(-1, 1)
        diff = np.abs(diff)
        diff = np.min(diff, axis=0)
        error = np.sum(pmf * diff)
        return error
    ###################################
    assert (len(refs) > 0)
    best_ref = None; best_error = np.inf
    for ref in refs:
        error = compute_error(counts, ref)
        if error < best_error:
            best_error = error
            best_ref = ref
    ###################################
    assert best_ref is not None
    best_ref.sort()
    return best_ref

####################################################






















