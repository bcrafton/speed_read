
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

from scipy.special import erf
def cdf(x, mu, sd):
    a = (x - mu) / (np.sqrt(2) * sd)
    return 0.5 * (1 + erf(a))

##########################################

eps = 1e-10
inf = 1e10

def expected_error(params, adc_count, row_count):

    ########################################################################

    nrow = np.maximum(1, row_count).reshape(8, 1, 1, params['max_rpr'], 1, 1)

    adc      = np.arange(params['adc'] + 1, dtype=np.float32)
    adc_low  = np.array([-inf, 0.2] + (adc[2:] - 0.5).tolist())
    adc_high = np.array([0.2]       + (adc[2:] - 0.5).tolist() + [inf])
    
    adc      =      adc.reshape(params['adc']+1, 1, 1, 1)
    adc_low  =  adc_low.reshape(params['adc']+1, 1, 1, 1)
    adc_high = adc_high.reshape(params['adc']+1, 1, 1, 1)

    ########################################################################

    RPR = params['max_rpr']
    
    N_lrs = np.zeros(shape=(RPR + 1, RPR + 1))
    N_hrs = np.zeros(shape=(RPR + 1, RPR + 1))

    for j in range(RPR + 1):
        for k in range(RPR + 1):
            N_lrs[j, k] = (k    ) if (k <= j) else 0
            N_hrs[j, k] = (j - k) if (k <= j) else 0

    ########################################################################

    print (np.shape(adc_count))
    p = adc_count[:, :, 1:, :, :].astype(np.float32)
    p = p.reshape(8, 8, 1, RPR, RPR + 1, RPR + 1)
    p = p / np.sum(p, axis=(4, 5), keepdims=True)
    assert np.allclose(np.sum(p, axis=(4, 5)), 1)

    ########################################################################

    mu  = N_lrs
    var = (params['lrs'] ** 2. * N_lrs) + (params['hrs'] ** 2. * N_hrs)
    sd  = np.sqrt(var)

    p_h = cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = cdf(adc_low, mu, np.maximum(sd, eps))
    pe = np.clip(p_h - p_l, 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)
    assert np.allclose(np.sum(pe, axis=0), 1)

    ########################################################################

    s = np.arange(0, params['max_rpr']+1, dtype=np.float32)
    e = adc - s

    assert np.allclose(np.sum(pe * p, axis=(2, 4, 5)), 1)

    error = p * pe * e * nrow
    mse  = np.sum(np.absolute(error), axis=(2, 4, 5))
    mean = np.sum(           (error), axis=(2, 4, 5))

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

    delay = row_count

    mse, mean = expected_error(params=params, adc_count=adc_count, row_count=row_count)

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

    assert (np.sum(error) >= np.sum(np.abs(mean)))
    return rpr_lut, bias_lut, np.sum(error), np.sum(mean)
    
    
##########################################
    
    

