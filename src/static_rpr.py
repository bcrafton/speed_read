
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr

import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def round_fraction(x, f):
    return np.around(x / f) * f

##########################################

def expected_error(params, adc_count, row_count, sat_count, rpr, nrow, bias):

    #######################
    # error from rpr <= adc
    #######################
    
    s  = np.arange(rpr + 1, dtype=np.float32)
    
    adc      = np.arange(params['adc'] + 1, dtype=np.float32).reshape(-1, 1)
    adc_low  = np.array([-1e6, 1e-6, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
    adc_high = np.array([1e-6, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)
    
    if rpr < params['adc']:
        adc_low[rpr+1:] = 1e6
        adc_high[rpr:] = 1e6

    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-9) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-9)
    e = adc - s
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-9)

    assert ( np.all(np.sum(pe, axis=0) == 1) )
    assert ( np.absolute(np.sum(pe * p) - 1) <= 1e-6 )
    if rpr < params['max_rpr']:
        assert ( np.sum(adc_count[rpr, rpr+1:]) == 0 )

    #######################
    # error from rpr > adc
    #######################
    
    mse = np.sum(np.absolute(p * pe * e * nrow))
    mean = np.sum(p * pe * e * nrow)

    # mse = ((np.sqrt(nrow) - 1) * mean + mse) / np.sqrt(nrow)
    return mse, mean

##########################################

def static_rpr(low, high, params, adc_count, row_count, sat_count, nrow, q, ratio):
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

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):

                #####################################################

                if rpr > params['adc']:
                    count = adc_count[xb, wb, rpr, sat_low:sat_high]
                    prob = count / (np.sum(count) + 1e-6)
                    weight = np.arange(sat_high - sat_low, dtype=np.float32)
                    bias = 0. # np.sum(prob * weight)
                else:
                    bias = 0.

                #####################################################

                total_row = max(1, row_count[xb][rpr - 1])

                scale = 2**wb * 2**xb
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], sat_count=sat_count[xb][wb], rpr=rpr, nrow=total_row, bias=bias)
                scaled_mse = (scale / q) * mse * ratio
                scaled_mean = (scale / q) * mean * ratio

                bias_table[xb][wb][rpr - 1] = bias
                error_table[xb][wb][rpr - 1] = scaled_mse
                mean_table[xb][wb][rpr - 1] = scaled_mean

                delay[xb][wb][rpr - 1] = row_count[xb][rpr - 1]

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
    error_table = round_fraction(error_table, 1e-4) - round_fraction(np.absolute(mean_table), 1e-4)
    mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)

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

    return rpr_lut, bias_lut
    
    
##########################################
    
    

