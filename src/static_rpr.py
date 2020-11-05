
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr

import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def expected_error(params, adc_count, row_count, rpr, nrow, bias):

    #######################
    # error from rpr <= adc
    #######################
    
    s  = np.arange(rpr + 1, dtype=np.float32)
    
    adc      = np.arange(params['adc'] + 1, dtype=np.float32).reshape(-1, 1)
    adc_low  = np.array([-1e6, 0.2, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
    adc_high = np.array([0.2, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)
    
    if rpr < params['adc']:
        adc_low[rpr+1:] = 1e6
        adc_high[rpr:] = 1e6

    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-9) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-9)
    e = s - adc
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-9)

    #######################
    # error from rpr > adc
    #######################
    
    if rpr > params['adc']:
        e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - bias
        # e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - round(nrow * bias) // nrow

    # mse = np.sum((p * pe * e * nrow) ** 2)
    # mse = np.sqrt(np.sum((p * pe * e * nrow) ** 2))
    # mse = np.sqrt(np.sum((p * pe * e) ** 2) * nrow)
    mse = np.sum(np.absolute(p * pe * e * nrow))

    mean = np.sum(p * pe * e * nrow)

    return mse, mean

##########################################

def static_rpr(low, high, params, adc_count, row_count, nrow, q):

    weight = np.arange(params['max_rpr']+1, dtype=np.float32)
    nrow_array = np.sum(row_count * weight, axis=2) / (np.sum(row_count, axis=2) + 1e-6)

    ############
    
    sat_low = params['adc']
    sat_high = high + 1

    ############

    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
    bias_lut = np.zeros(shape=(8, 8), dtype=np.float32)
    mse_lut = np.zeros(shape=(8, 8), dtype=np.float32)
    mean_lut = np.zeros(shape=(8, 8), dtype=np.float32)

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']
        
    if not (params['skip'] and params['cards']):
        return rpr_lut, bias_lut

    delay      = np.zeros(shape=(8, 8, 64))
    error      = np.zeros(shape=(8, 8, 64))
    bias_table = np.zeros(shape=(8, 8, 64))

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):

                #####################################################

                if rpr > params['adc']:
                    count = adc_count[xb, wb, rpr, sat_low:sat_high]
                    prob = count / (np.sum(count) + 1e-6)
                    weight = np.arange(sat_high - sat_low, dtype=np.float32)
                    bias = np.sum(prob * weight)
                else:
                    bias = 0.

                #####################################################

                # total_row = np.ceil(nrow / params['wl']) * np.ceil(nrow_array[xb][rpr])
                total_row = np.ceil(nrow / params['wl'] * nrow_array[xb][rpr])
                # total_row = nrow / params['wl'] * nrow_array[xb][rpr]

                scale = 2**wb * 2**xb
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], rpr=rpr, nrow=total_row, bias=bias)
                scaled_mse = (scale / q) * mse
                scaled_mean = (scale / q) * mean

                bias_table[xb][wb][rpr - 1] = bias
                error[xb][wb][rpr - 1] = scaled_mse
                delay[xb][wb][rpr - 1] = nrow_array[xb][rpr]

    # print (np.min(error, axis=2))
    # print ()
    # print (np.max(error, axis=2))

    # print (np.sum(np.min(error, axis=2)))
    # print (np.sum(np.max(error, axis=2)))

    assert (np.sum(np.min(error, axis=2)) <= params['thresh'])

    # KeyError: 'infeasible problem'
    # https://stackoverflow.com/questions/46246349/infeasible-solution-for-an-lp-even-though-there-exists-feasible-solutionusing-c
    # need to clip precision.
    error = np.clip(error, 1e-6, np.inf)

    rpr_lut = optimize_rpr(error, delay, params['thresh'])
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr = rpr_lut[xb][wb]
            bias_lut[xb][wb] = bias_table[xb][wb][rpr - 1]

    return rpr_lut, bias_lut
    
    
##########################################
    
    

