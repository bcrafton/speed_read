
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

    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']
            
    ##############################################

    adc_state = np.zeros(shape=(params['adc'], params['adc'], params['adc'] + 1))
    adc_thresh = np.zeros(shape=(params['adc'], params['adc'], params['adc'] + 1))

    weight = np.arange(65, dtype=np.float32)
    nrow_array = np.sum(row_count * weight, axis=2) / (np.sum(row_count, axis=2) + 1e-6)
    nrow_array = np.ceil(nrow_array)
    
    expected_cycles = np.ceil(nrow / params['wl']) * np.ceil(nrow_array)

    ##############################################

    for xb in range(params['bpa']):
        for wb in range(params['bpw']):
            for rpr in range(low, high + 1):
            
                # print (xb, wb, rpr)
            
                counts = adc_count[xb][wb][rpr][0:rpr+1]
                values = np.array(range(rpr+1))
                prob = counts / np.sum(counts)
                
                if rpr <= params['adc']:
                    centroids = np.arange(0, params['adc'] + 1, step=1, dtype=np.float32)
                elif np.count_nonzero(counts) <= params['adc']:
                    centroids = np.arange(0, params['adc'] + 1, step=1, dtype=np.float32)
                else:
                    centroids = sorted(kmeans(values=values, counts=counts, n_clusters=params['adc'] + 1))

                mse = exp_err(s=values, p=prob, var=params['sigma'], adc=centroids, rpr=rpr, row=expected_cycles[xb][rpr])
                scale = 2**wb * 2**xb
                scaled_mse = (scale / q) * 64. * mse
                
                if (rpr == low) or (scaled_mse < params['thresh']):
                    rpr_lut[xb][wb] = rpr
                    adc_state[xb][wb] = 4 * np.array(centroids)
                    adc_thresh[xb][wb] = adc_floor(centroids)
                    if rpr == 1: adc_thresh[xb][wb][0] = 0.2
                else:
                    break

    return rpr_lut, adc_state, adc_thresh
        
#########################







        
        
        
