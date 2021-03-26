
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

def expected_error(params, adc_count, nrow):

    ########################################################################

    nrow = np.reshape(nrow, (params['max_rpr'], 1, 1))

    adc      = np.arange(params['adc'] + 1, dtype=np.float32)
    adc_low  = np.array([-inf, 0.2] + (adc[2:] - 0.5).tolist())
    adc_high = np.array([0.2]       + (adc[2:] - 0.5).tolist() + [inf])

    adc_low  = np.tile( adc_low.reshape(-1, 1), reps=[1, params['max_rpr']])
    adc_high = np.tile(adc_high.reshape(-1, 1), reps=[1, params['max_rpr']])

    for rpr in range(0, params['adc']):
        adc_low [rpr+2:, rpr] = inf
        adc_high[rpr+1:, rpr] = inf

    adc      =      adc.reshape(params['adc']+1,                 1, 1, 1)
    adc_low  =  adc_low.reshape(params['adc']+1, params['max_rpr'], 1, 1)
    adc_high = adc_high.reshape(params['adc']+1, params['max_rpr'], 1, 1)

    ########################################################################

    RPR = params['max_rpr']

    N_lrs = np.zeros(shape=(RPR, RPR + 1, RPR + 1))
    N_hrs = np.zeros(shape=(RPR, RPR + 1, RPR + 1))
    for i in range(RPR):
        for j in range(RPR + 1):
            for k in range(RPR + 1):
                N_lrs[i, j, k] = (k    ) if (k <= j and j <= i+1) else 0
                N_hrs[i, j, k] = (j - k) if (k <= j and j <= i+1) else 0

    ########################################################################

    p = adc_count[1:].astype(np.float32)
    p = p / np.sum(p, axis=(1, 2), keepdims=True)

    ########################################################################

    mu  = N_lrs
    var = (params['lrs'] ** 2. * N_lrs) + (params['hrs'] ** 2. * N_hrs)
    sd  = np.sqrt(var)

    p_h = norm.cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = norm.cdf(adc_low, mu, np.maximum(sd, eps))
    pe = np.clip(p_h - p_l, 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)

    ########################################################################

    s = np.arange(0, params['max_rpr']+1, dtype=np.float32)
    e = adc - s

    assert np.allclose(np.sum(p,      axis=(1, 2)),    1)
    assert np.allclose(np.sum(pe,     axis=0),         1)
    assert np.allclose(np.sum(pe * p, axis=(0, 2, 3)), 1)

    mse  = np.sum(np.absolute(p * pe * e * nrow), axis=(0, 2, 3))
    mean = np.sum(           (p * pe * e * nrow), axis=(0, 2, 3))
    return mse, mean

##########################################

def static_rpr(low, high, params, adc_count, row_count, nrow, q, ratio):
    assert (q > 0)

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

            total_row = np.maximum(1, row_count[xb])
            delay[xb][wb] = row_count[xb]

            mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], nrow=total_row)
            assert np.all(mse >= np.abs(mean))
            # print (np.around(mse))
            # print (mean)
            
            scale = 2**wb * 2**xb / q * ratio
            error_table[xb][wb] = scale * mse
            mean_table[xb][wb] = scale * mean

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
    
    

