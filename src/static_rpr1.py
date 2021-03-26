
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr

import time
import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def round_fraction(x, f):
    return np.around(x / f) * f

##########################################

eps = 1e-10
inf = 1e10

def expected_error(params, adc_count, row_count, rpr, nrow, bias):

    ########################################################################

    adc      = np.arange(params['adc'] + 1, dtype=np.float32)
    adc_low  = np.array([-inf, 0.2] + (adc[2:] - 0.5).tolist())
    adc_high = np.array([0.2]       + (adc[2:] - 0.5).tolist() + [inf])

    adc_low  = np.tile( adc_low.reshape(-1, 1), reps=[1, params['max_rpr']+1])
    adc_high = np.tile(adc_high.reshape(-1, 1), reps=[1, params['max_rpr']+1])

    for rpr in range(0, params['adc']):
        adc_low[rpr + 1:] = inf
        adc_high[rpr:] = inf

    adc      = adc.reshape(-1, 1, 1, 1)
    adc_low  = adc_low.reshape(params['adc']+1, params['max_rpr']+1, 1, 1)
    adc_high = adc_high.reshape(params['adc']+1, params['max_rpr']+1, 1, 1)

    ########################################################################

    def row(S, E, N):
        if S < E: return np.append( np.arange(S, E+1,  1), [0] * (N - 1 - E + S) )
        else:     return np.append( np.arange(S, E-1, -1), [0] * (N - 1 - S + E) )

    N_lrs = np.stack([row(0, i, params['max_rpr']+1) for i in range(params['max_rpr']+1)], axis=0)
    N_hrs = np.stack([row(i, 0, params['max_rpr']+1) for i in range(params['max_rpr']+1)], axis=0)

    ########################################################################

    p = adc_count.astype(np.float32)
    p[1:] = p[1:] / np.sum(p[1:], axis=(1, 2), keepdims=True)

    ########################################################################

    mu  = N_lrs
    var = (params['lrs'] ** 2. * N_lrs) + (params['hrs'] ** 2. * N_hrs)
    sd  = np.sqrt(var)

    p_h = norm.cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = norm.cdf(adc_low, mu, np.maximum(sd, eps))
    pe = np.clip(p_h - p_l, 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)

    ########################################################################

    s = np.arange(params['max_rpr']+1, dtype=np.float32)    
    e = adc - s

    assert np.allclose(np.sum(pe, axis=0), 1)
    assert np.allclose(np.sum(pe[:, 1:] * p[1:], axis=(0, 2, 3)), 1)

    mse = np.sum(np.absolute(p * pe * e * nrow))
    mean = np.sum(p * pe * e * nrow)
    return mse, mean

##########################################

def static_rpr(low, high, params, adc_count, row_count, nrow, q, ratio):
    assert (q > 0)

    ############

    sat_low = params['adc']
    sat_high = high + 1

    ############

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']
    bias_lut = np.zeros(shape=(8, 8), dtype=np.float32)

    delay       = np.zeros(shape=(8, 8, high))
    error_table = np.zeros(shape=(8, 8, high))
    mean_table = np.zeros(shape=(8, 8, high))
    bias_table  = np.zeros(shape=(8, 8, high))

    start = time.time()
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):

            total_row = np.maximum(1, row_count[xb][rpr - 1])

            scale = 2**wb * 2**xb

            mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], rpr=rpr, nrow=total_row, bias=None)
            scaled_mse = (scale / q) * mse * ratio
            scaled_mean = (scale / q) * mean * ratio

            assert (scaled_mse >= np.abs(scaled_mean))

            error_table[xb][wb][rpr - 1] = scaled_mse
            mean_table[xb][wb][rpr - 1] = scaled_mean

            delay[xb][wb][rpr - 1] = row_count[xb][rpr - 1]
            # if params['SAR']: 
            if True:
                sar = max(1, np.ceil(np.log2(min(rpr, params['adc']))))
                delay[xb][wb][rpr - 1] *= sar
                

    assert (np.sum(mean_table[:, :, 0]) >= -params['thresh'])
    assert (np.sum(mean_table[:, :, 0]) <=  params['thresh'])
    assert (np.sum(np.min(error_table, axis=2)) <= params['thresh'])

    # KeyError: 'infeasible problem'
    # https://stackoverflow.com/questions/46246349/infeasible-solution-for-an-lp-even-though-there-exists-feasible-solutionusing-c
    # need to clip precision.
    #
    # error_table = np.clip(error_table, 1e-6, np.inf) - np.clip(np.absolute(mean_table), 1e-6, np.inf)
    # mean_table = np.sign(mean_table) * np.clip(np.absolute(mean_table), 1e-6, np.inf)
    # 
    # error_table = round_fraction(error_table, 1e-4) - round_fraction(np.absolute(mean_table), 1e-4)
    # mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)
    # 
    error_table = round_fraction(error_table, 1e-4)
    mean_table = round_fraction(mean_table, 1e-4)

    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    cycle = np.zeros(shape=(8, 8))

    print (time.time() - start)
    start = time.time()

    if params['skip'] and params['cards']:
        rpr_lut = optimize_rpr(error_table, mean_table, delay, params['thresh'])
        for wb in range(params['bpw']):
            for xb in range(params['bpa']):
                rpr = rpr_lut[xb][wb]
                bias_lut[xb][wb] = bias_table[xb][wb][rpr - 1]

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr = rpr_lut[xb][wb]
            error[xb][wb] = error_table[xb][wb][rpr-1]
            mean[xb][wb] = mean_table[xb][wb][rpr-1]
            cycle[xb][wb] = delay[xb][wb][rpr-1]

    # print (np.around(error, 2))
    # print (np.around(mean, 2))
    assert (np.sum(error) >= np.sum(np.abs(mean)))
    print (rpr_lut)
    print (time.time() - start)
    
    return rpr_lut, bias_lut, np.sum(error), np.sum(mean)
    
    
##########################################
    
    

