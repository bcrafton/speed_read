
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr

import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def expected_error(params, adc_count, row_count, sat_count, rpr, nrow, bias):

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
    e = adc - s
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-9)

    assert ( np.all(np.sum(pe, axis=0) == 1) )
    assert ( np.absolute(np.sum(pe * p) - 1) <= 1e-6 )
    if rpr < params['max_rpr']:
        assert ( np.sum(adc_count[rpr, rpr+1:]) == 0 )

    # print (rpr)
    # print (np.around(e * pe, 1))
    # print (np.around(pe * p, 2))
    # print (np.around(pe, 2))

    '''
    if rpr == 16:
        print (np.around(pe * p, 2))
        assert (False)
    '''

    #######################
    # error from rpr > adc
    #######################

    # sat_count = sat_count[rpr][1:rpr+1]
    # sat_pmf = sat_count / (np.sum(sat_count) + 1e-9)
    # sat = np.arange(1, rpr+1, dtype=np.float32)

    # why are we changing e here ...
    '''
    if rpr > params['adc']:
        e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - bias
        # e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - round(bias * np.sum(sat_pmf * sat))
        # e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - round(nrow * bias) // nrow
    '''

    

    # mse = np.sum((p * pe * e * nrow) ** 2)
    # mse = np.sqrt(np.sum((p * pe * e * nrow) ** 2))
    # mse = np.sqrt(np.sum((p * pe * e) ** 2) * nrow)
    # mse = np.sum(np.absolute(p * pe * e * nrow))
    
    mse = np.sum(np.absolute(p * pe * e * nrow))
    mean = np.sum(p * pe * e * nrow)
    
    # mse = mse - (mse - np.absolute(mean)) # / np.sqrt(2)
    # mse = np.absolute(mean)
    # mse = (mse + np.absolute(mean)) / 2
    
    return mse, mean, np.sum(np.absolute(p * pe * e))

##########################################

def static_rpr(low, high, params, adc_count, row_count, sat_count, nrow, q, ratio):
    '''
    if q < 1:
        assert (q == 0)
        q = 1
    '''
    assert (q > 0)

    # weight = np.arange(params['wl'] + 1, dtype=np.float32)
    # nrow_array = np.sum(row_count * weight, axis=2) / (np.sum(row_count, axis=2) + 1e-6)

    ############
    
    sat_low = params['adc']
    sat_high = high + 1

    ############

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']
    bias_lut = np.zeros(shape=(8, 8), dtype=np.float32)
    mse_lut = np.zeros(shape=(8, 8), dtype=np.float32)
    mean_lut = np.zeros(shape=(8, 8), dtype=np.float32)

    delay       = np.zeros(shape=(8, 8, high))
    error_table = np.zeros(shape=(8, 8, high))
    mean_table = np.zeros(shape=(8, 8, high))
    bias_table  = np.zeros(shape=(8, 8, high))
    ppee_table  = np.zeros(shape=(8, 8, high))

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

                # total_row = np.ceil(nrow / params['wl']) * np.ceil(nrow_array[xb][rpr])
                # total_row = np.ceil(nrow / params['wl'] * nrow_array[xb][rpr])
                total_row = max(1, row_count[xb][rpr - 1])

                # print (xb, wb, rpr, total_row)

                scale = 2**wb * 2**xb
                mse, mean, ppee = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], sat_count=sat_count[xb][wb], rpr=rpr, nrow=total_row, bias=bias)
                scaled_mse = (scale / q) * mse * ratio
                scaled_mean = scale * mean

                bias_table[xb][wb][rpr - 1] = bias
                error_table[xb][wb][rpr - 1] = scaled_mse
                mean_table[xb][wb][rpr - 1] = scaled_mean

                ppee_table[xb][wb][rpr - 1] = (scale / q) * ppee
                delay[xb][wb][rpr - 1] = row_count[xb][rpr - 1]

    assert (np.sum(np.min(error_table, axis=2)) <= params['thresh'])

    # KeyError: 'infeasible problem'
    # https://stackoverflow.com/questions/46246349/infeasible-solution-for-an-lp-even-though-there-exists-feasible-solutionusing-c
    # need to clip precision.
    error_table = np.clip(error_table, 1e-9, np.inf)

    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    cycle = np.zeros(shape=(8, 8))
    row = np.zeros(shape=(8, 8))
    ppee = np.zeros(shape=(8, 8))

    if params['skip'] and params['cards']:
        rpr_lut = optimize_rpr(error_table, delay, params['thresh'])
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
            row[xb][wb] = row_count[xb][rpr - 1]
            ppee[xb][wb] = ppee_table[xb][wb][rpr - 1]

    # print (np.around(error / np.sum(error), 2))
    # print (np.sum(error))
    
    # print (np.around(ppee, 3))
    # print (np.around(row, 1))

    # assert (False)
    # print (np.sum(error), np.sum(mean))
    
    return rpr_lut, bias_lut, np.sum(error), row
    
    
##########################################
    
    

