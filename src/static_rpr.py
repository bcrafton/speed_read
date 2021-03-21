
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr

import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def round_fraction(x, f):
    return np.around(x / f) * f

##########################################

eps = 1e-10
inf = 1e10

def expected_error(params, adc_count, row_count, rpr, nrow, bias):

    ratio, lrs, hrs = params['sigma']

    s = np.arange(rpr + 1, dtype=np.float32)
    
    adc      = np.arange(params['adc'] + 1, dtype=np.float32)
    adc_low  = np.array([-inf, 0.2] + (adc[2:] - 0.5).tolist())
    adc_high = np.array([0.2]       + (adc[2:] - 0.5).tolist() + [inf])

    if rpr < params['adc']:
        adc_low[rpr + 1:] = inf
        adc_high[rpr:] = inf

    adc      = adc.reshape(-1, 1, 1)
    adc_low  = adc_low.reshape(-1, 1, 1)
    adc_high = adc_high.reshape(-1, 1, 1)

    on_counts  = adc_count[rpr, 0:(rpr + 1), 0:(rpr + 1)]
    off_counts = np.flip(on_counts, axis=1)

    def row(S, E, N):
        if S < E: return np.append( np.arange(S, E+1,  1), [0] * (N - 1 - E + S) )
        else:     return np.append( np.arange(S, E-1, -1), [0] * (N - 1 - S + E) )

    N_lrs = np.stack([row(0, i, rpr + 1) for i in range(rpr + 1)], axis=0)
    N_hrs = np.stack([row(i, 0, rpr + 1) for i in range(rpr + 1)], axis=0)
    
    e = adc - s
    p = on_counts / np.sum(on_counts)

    '''
    if rpr == 16:
        print (np.around(p, 2))
        print (np.around(np.sum(p, axis=1), 2))
        print ()
    '''

    mu  = N_lrs
    var = (lrs ** 2. * N_lrs) + ((hrs / ratio) ** 2. * N_hrs)
    sd  = np.sqrt(var)

    # before it would be [17x2] ... now its [17x2x2]
    # pe along [17] should be 1 right ?
    # 17 different ADC thresholds should sum to 1.

    p_h = norm.cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = norm.cdf(adc_low, mu, np.maximum(sd, eps))
    # pe = np.clip(p_h - p_l - 2 * norm.cdf(-3), 0, 1)
    # why 3.4 ? 
    # because we approximate normal cdf in C, and the approximation actually sucks
    # we should probably do this based off of that distribution to tell us if other errors exist ...
    # like literally take the lut_var in here.
    # pe = np.clip(p_h - p_l - 2 * norm.cdf(-3.4), 0, 1)
    pe = np.clip(p_h - p_l - 1 * norm.cdf(-3), 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)

    assert (np.allclose(np.sum(pe, axis=0), 1))
    assert (np.allclose(np.sum(pe * p), 1))
    
    # print (np.shape(p))  #    8x8
    # print (np.shape(pe)) # 17x8x8
    # print (np.shape(e))  # 17x1x8
    
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

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):
                bias = 0.

                total_row = max(1, row_count[xb][rpr - 1])

                scale = 2**wb * 2**xb
                # if rpr == 16: print (xb, wb, end=' ')
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], rpr=rpr, nrow=total_row, bias=bias)
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

    return rpr_lut, bias_lut, np.sum(error), np.sum(mean)
    
    
##########################################
    
    

