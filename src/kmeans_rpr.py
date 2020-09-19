
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, binom
from kmeans import kmeans

#########################

def adc_range(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_low = np.zeros_like(adc, dtype=np.float32)
    adc_high = np.zeros_like(adc, dtype=np.float32)
    
    adc_low[0] = -1e2
    adc_high[-1] = 1e2
    
    for s in range(len(adc) - 1):
        adc_high[s] = (adc[s] + adc[s + 1]) / 2
        adc_low[s + 1] = (adc[s] + adc[s + 1]) / 2

    return adc_low, adc_high
    
#########################

def adc_floor(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_thresh = np.zeros_like(adc, dtype=np.float32)
    
    for s in range(len(adc) - 1):
        adc_thresh[s] = (adc[s] + adc[s + 1]) / 2

    adc_thresh[-1] = adc[-1]
    
    return adc_thresh

#########################

def exp_err(s, p, var, adc, rpr, row):
    assert (np.all(p <= 1.))
    assert (len(s) == len(p))

    adc = sorted(adc)
    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    pe = norm.cdf(adc_high, s, var * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, var * np.sqrt(s) + 1e-6)
    e = s - adc

    mse = np.sum(np.absolute(p * pe * e * row))
    return mse

#########################

def kmeans_rpr(low, high, params, adc_count, row_count, nrow, q):

    adc_state = np.zeros(shape=(high + 1, params['adc'] + 1))
    adc_thresh = np.zeros(shape=(high + 1, params['adc'] + 1))

    weight = np.arange(65, dtype=np.float32)
    nrow_array = np.sum(row_count * weight, axis=2) / (np.sum(row_count, axis=2) + 1e-6)
    nrow_array = np.mean(nrow_array, axis=0)
    nrow_array = np.ceil(nrow_array)
    
    expected_cycles = np.ceil(nrow / params['wl']) * np.ceil(nrow_array)

    rpr_dist = {}
    for rpr in range(low, high + 1):
        counts = np.sum(adc_count, axis=(0, 1))[rpr][0:rpr+1]
        values = np.array(range(rpr+1))
        
        if rpr <= params['adc']:
            centroids = np.arange(0, params['adc'] + 1, step=1, dtype=np.float32)
        else:
            centroids = kmeans(values=values, counts=counts, n_clusters=params['adc'] + 1)
            centroids = sorted(centroids)
        
        p = counts / np.sum(counts)
        s = values

        mse = exp_err(s=s, p=p, var=params['sigma'], adc=centroids, rpr=rpr, row=expected_cycles[rpr])
        rpr_dist[rpr] = {'mse': mse, 'centroids': centroids}
        
        adc_state[rpr] = 4 * np.array(centroids)
        adc_thresh[rpr] = adc_floor(centroids)
        
        if rpr == 1:
            adc_thresh[rpr][0] = 0.2
        
    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']
    
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):
            
                scale = 2**wb * 2**xb
                mse = rpr_dist[rpr]['mse']
                scaled_mse = (scale / q) * 64. * mse
                
                if rpr == low:
                    rpr_lut[xb][wb] = rpr
                elif scaled_mse < params['thresh']:
                    rpr_lut[xb][wb] = rpr

    return rpr_lut, adc_state, adc_thresh
        
#########################







        
        
        
