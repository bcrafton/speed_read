
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

from kmeans import kmeans
from optimize_rpr import optimize_rpr

##########################################

def round_fraction(x, f):
    return np.around(x / f) * f

#########################

def adc_range(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_low = np.zeros_like(adc, dtype=np.float32)
    adc_high = np.zeros_like(adc, dtype=np.float32)
    
    adc_low[0] = -1e6
    adc_high[-1] = 1e6
    
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

def expected_error(params, adc_count, row_count, centroids, rpr, nrow):
    s = np.arange(rpr + 1, dtype=np.float32)

    adc = sorted(centroids)
    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    if rpr < params['adc']:
        adc_low[1] = 1e-6
        adc_high[0] = 1e-6
        adc_low[rpr+1:] = 1e6
        adc_high[rpr:] = 1e6

    # verify s=0 produces no errors. when 1e-6 was 1e-9, we saw s=0 produce errors.
    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-9) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-9)
    e = adc - s
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-9)

    assert(        np.absolute(1. - np.sum(p))          < 1e-6  )
    assert( np.all(np.absolute(1. - np.sum(pe, axis=0)) < 1e-6) )

    mse = np.sum(np.absolute(p * pe * e * nrow))
    mean = np.sum(p * pe * e * nrow)

    return mse, mean

#########################

def kmeans_rpr(low, high, params, adc_count, row_count, nrow, q, ratio):
    assert (q > 0)

    ##############################################

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']

    ##############################################

    delay       = np.zeros(shape=(8, 8, high))
    error_table = np.zeros(shape=(8, 8, high))
    mean_table  = np.zeros(shape=(8, 8, high))

    centroids_table = np.zeros(shape=(8, 8, high, params['adc'] + 1))

    ##############################################

    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']

    ##############################################

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):

                counts = np.sum(adc_count, axis=(0, 1))[rpr][0:rpr+1]
                values = np.array(range(rpr+1))
                probs = counts / np.sum(counts)
                
                if rpr <= params['adc']: centroids = np.arange(0, params['adc'] + 1, step=1, dtype=np.float32)
                else:                    centroids = sorted(kmeans(values=values, counts=counts, n_clusters=params['adc'] + 1))

                total_row = max(1, row_count[xb][rpr - 1])

                scale = 2**wb * 2**xb
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], centroids=centroids, rpr=rpr, nrow=total_row)
                scaled_mse = (scale / q) * mse * ratio
                scaled_mean = (scale / q) * mean * ratio

                error_table[xb][wb][rpr - 1] = scaled_mse
                mean_table[xb][wb][rpr - 1] = scaled_mean
                delay[xb][wb][rpr - 1] = row_count[xb][rpr - 1]
                centroids_table[xb][wb][rpr - 1] = centroids

    #########################

    error_table = round_fraction(error_table, 1e-4) - round_fraction(np.absolute(mean_table), 1e-4)
    mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)
    delay = round_fraction(delay, 1e-1)

    assert (np.sum(mean_table[:, :, 0]) >= -params['thresh'])
    assert (np.sum(mean_table[:, :, 0]) <=  params['thresh'])
    assert (np.sum(np.min(error_table, axis=2)) <= params['thresh'])

    adc_state  = np.zeros(shape=(8, 8, params['adc'] + 1))
    adc_thresh = np.zeros(shape=(8, 8, params['adc'] + 1))
    if params['skip'] and params['cards']:
        rpr_lut = optimize_rpr(error_table, mean_table, delay, params['thresh'])
        for wb in range(params['bpw']):
            for xb in range(params['bpa']):
                rpr = rpr_lut[xb][wb]
                adc_state[xb][wb] = 4 * np.array(centroids_table[xb][wb][rpr - 1])
                adc_thresh[xb][wb] = adc_floor(centroids_table[xb][wb][rpr - 1])
                if rpr == 1:
                    adc_thresh[xb][wb][0] = 0.2

    #########################

    '''
    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    cycle = np.zeros(shape=(8, 8))
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr = rpr_lut[xb][wb]
            error[xb][wb] = error_table[xb][wb][rpr-1]
            mean[xb][wb] = mean_table[xb][wb][rpr-1]
            cycle[xb][wb] = delay[xb][wb][rpr-1]
    '''

    #########################

    return rpr_lut, adc_state, adc_thresh
        








        
        
        
