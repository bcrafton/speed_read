
import numpy as np
from scipy.stats import norm, binom

from optimize_rpr import optimize_rpr
from adc import *

import time
import sys
np.set_printoptions(threshold=sys.maxsize)

##########################################

def round_fraction(x, f):
    return np.around(x / f) * f

##########################################

eps = 1e-10
inf = 1e10

def expected_error(params, step, rpr, adc, profile, row, adc_thresh, adc_value):

    # pe

    # step is passed as (2 ** step)
    adc_center =       adc_value.reshape(1 + min(rpr, adc) // step, 1, 1)
    adc_low    = adc_thresh[:-1].reshape(1 + min(rpr, adc) // step, 1, 1)
    adc_high   = adc_thresh[ 1:].reshape(1 + min(rpr, adc) // step, 1, 1)

    ########################################################################

    N_lrs = np.zeros(shape=(params['max_rpr'] + 1, params['max_rpr'] + 1))
    N_hrs = np.zeros(shape=(params['max_rpr'] + 1, params['max_rpr'] + 1))

    for j in range(params['max_rpr'] + 1):
        for k in range(params['max_rpr'] + 1):
            N_lrs[j, k] = (k    ) if (k <= j) else 0
            N_hrs[j, k] = (j - k) if (k <= j) else 0

    ########################################################################

    mu  = N_lrs
    var = (params['lrs'] ** 2. * N_lrs) + (params['hrs'] ** 2. * N_hrs)
    sd  = np.sqrt(var)

    p_h = norm.cdf(adc_high, mu, np.maximum(sd, eps))
    p_l = norm.cdf(adc_low, mu, np.maximum(sd, eps))
    pe = np.clip(p_h - p_l, 0, 1)
    pe = pe / np.sum(pe, axis=0, keepdims=True)

    ########################################################################

    # p
    p = profile.astype(np.float32)
    p = p / np.sum(p)

    ########################################################################

    # e
    s = np.arange(0, params['max_rpr'] + 1, dtype=np.float32)
    e = adc_center - s
    
    ########################################################################

    assert np.allclose(np.sum(p             ), 1)
    assert np.allclose(np.sum(pe,     axis=0), 1)
    assert np.allclose(np.sum(pe * p        ), 1)

    error = p * pe * e * row
    mse  = np.sum(np.absolute(error))
    mean = np.sum(           (error))

    return mse, mean

##########################################

def static_rpr(id, params, q):
    assert (q > 0)

    ############

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']
    step_lut = np.zeros(shape=(8, 8), dtype=np.int32)

    ############

    delay_table = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], params['adc'], 2))
    error_table = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], params['adc'], 2))
    mean_table  = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], params['adc'], 2))
    valid_table = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], params['adc'], 2))
    area_table  = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], params['adc'], 2))

    thresh_table = {}
    value_table = {}

    profile = np.load('./profile/%d.npy' % (id), allow_pickle=True).item()
    row = np.maximum(1, profile['row'])
    ratio = profile['ratio']
    profile = profile['adc']

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for step in range(params['max_step']):
                for rpr in range(params['max_rpr']):
                    for adc in range(params['adc']):
                        for sar in [0, 1]:
                            thresh, values = thresholds(profile[xb, wb, rpr + 1], 2 ** step, min(rpr + 1, adc + 1) // 2 ** step, method=params['method'])

                            thresh_table[(xb, wb, step, rpr, adc, sar)] = thresh
                            value_table[(xb, wb, step, rpr, adc, sar)] = values

                            mse, mean = expected_error(params, 2 ** step, rpr + 1, adc + 1, profile[xb, wb, rpr + 1], row[xb][rpr], thresh, values)
                            assert np.all(mse >= np.abs(mean))

                            scale = 2**wb * 2**xb / q * ratio
                            error_table[xb][wb][step][rpr][adc][sar] = scale * mse
                            mean_table [xb][wb][step][rpr][adc][sar] = scale * mean

                            if sar:
                                valid_table[xb][wb][step][rpr][adc][sar] = 1 if (min(rpr + 1, adc + 1) >= 2 ** step) else 0
                                cycle = min(rpr + 1, adc + 1)
                                cycle = 1 + np.ceil(np.log2(cycle)) - step
                                delay_table[xb][wb][step][rpr][adc][sar] = row[xb][rpr] * cycle
                                area_table[xb][wb][step][rpr][adc][sar] = 2
                            else:
                                valid_table[xb][wb][step][rpr][adc][sar] = 1 if (step == 0) else 0
                                delay_table[xb][wb][step][rpr][adc][sar] = row[xb][rpr]
                                area_table[xb][wb][step][rpr][adc][sar] = (adc + 1) // 2 ** step

    assert (np.sum(mean_table[:, :, 0, 0]) >= -params['thresh'])
    assert (np.sum(mean_table[:, :, 0, 0]) <=  params['thresh'])
    assert (np.sum(np.min(error_table, axis=(2, 3))) <= params['thresh'])

    # KeyError: 'infeasible problem'
    # https://stackoverflow.com/questions/46246349/infeasible-solution-for-an-lp-even-though-there-exists-feasible-solutionusing-c
    # need to clip precision.
    #
    # error_table = np.clip(error_table, 1e-6, np.inf) - np.clip(np.absolute(mean_table), 1e-6, np.inf)
    # mean_table = np.sign(mean_table) * np.clip(np.absolute(mean_table), 1e-6, np.inf)
    # 
    # this is silly. idky this works.
    # answer should be scaling by: [np.sqrt(nrow), nrow]
    # error should scale by sqrt(nrow)
    # mean should scale by nrow
    error_table = round_fraction(error_table, 1e-4) - round_fraction(np.absolute(mean_table), 1e-4)
    mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)
    # 
    # error_table = round_fraction(error_table, 1e-4)
    # mean_table = round_fraction(mean_table, 1e-4)

    if params['skip'] and params['cards']:
        rpr_lut, step_lut, sar_lut, adc_lut, num_lut = optimize_rpr(error_table, np.abs(mean_table), delay_table, area_table, valid_table, params['thresh'])

    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    conf = np.zeros(shape=(8, 8, 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']), dtype=np.uint32)
    value = np.zeros(shape=(8, 8, 1 + params['adc']), dtype=np.float32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            step = step_lut[xb][wb] 
            rpr  = rpr_lut[xb][wb] - 1
            adc  = adc_lut[xb][wb] - 1
            sar  = sar_lut[xb][wb]

            error[xb][wb] = error_table[xb][wb][step][rpr][adc][sar]
            mean[xb][wb]  = mean_table[xb][wb][step][rpr][adc][sar]

            conf1 = confusion(thresh_table[(xb, wb, step, rpr, adc, sar)], params['max_rpr'], min(rpr + 1, adc + 1) // 2 ** step, params['hrs'], params['lrs'])
            conf2 = np.zeros(shape=(1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] - min(rpr + 1, adc + 1) // 2 ** step), dtype=np.uint32)
            conf[xb][wb] = np.concatenate((conf1, conf2), axis=-1)

            values1 = value_table[(xb, wb, step, rpr, adc, sar)]
            values2 = -1 * np.ones(shape=(params['adc'] - min(rpr + 1, adc + 1) // 2 ** step), dtype=np.float32)
            value[xb][wb] = np.concatenate((values1, values2), axis=-1)

            # print ( thresh_table[(xb, wb, step, rpr, adc, sar)] )
            # print (  value_table[(xb, wb, step, rpr, adc, sar)] )

    return rpr_lut, step_lut, sar_lut, conf, value

##########################################
    
    

