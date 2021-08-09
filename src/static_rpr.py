
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
'''
from scipy.special import erf
def cdf(x, mu, sd):
    a = (x - mu) / (np.sqrt(2) * sd)
    return 0.5 * (1 + erf(a))
'''
##########################################

eps = 1e-10
inf = 1e10

def expected_error(params, rpr, step, adc_hist, row, adc_thresh, adc_value):

    # pe

    # step is passed as (2 ** step)
    adc      =       adc_value.reshape(1 + min(rpr, params['adc']) // step, 1, 1)
    adc_low  = adc_thresh[:-1].reshape(1 + min(rpr, params['adc']) // step, 1, 1)
    adc_high = adc_thresh[ 1:].reshape(1 + min(rpr, params['adc']) // step, 1, 1)

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
    p = adc_hist.astype(np.float32)
    p = p / np.sum(p)

    ########################################################################

    # e
    s = np.arange(0, params['max_rpr']+1, dtype=np.float32)
    e = adc - s
    
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

    delay       = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr']))
    error_table = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr']))
    mean_table  = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr']))

    # [xb, wb] [step, rpr] [wl] [on, adc]
    conf_table  = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'], 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']))
    # [xb, wb] [step, rpr] [adc]
    value_table = np.zeros(shape=(8, 8, params['max_step'], params['max_rpr'],                                               1 + params['adc']))

    profile = np.load('./profile/%d.npy' % (id), allow_pickle=True).item()
    adc = profile['adc']
    row = np.maximum(1, profile['row'])
    ratio = profile['ratio']

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(params['max_rpr']):
                for step in range(params['max_step']):

                    thresh, values = thresholds(adc[xb, wb, rpr + 1], min(rpr + 1, params['adc']) // 2 ** step, 2 ** step, method=params['method'])
                    conf = confusion(thresh, params['max_rpr'], min(rpr + 1, params['adc']) // 2 ** step, params['hrs'], params['lrs'])

                    mse, mean = expected_error(params, rpr + 1, 2 ** step, adc[xb, wb, rpr + 1], row[xb][rpr], thresh, values)
                    assert np.all(mse >= np.abs(mean))

                    conf_pad =       np.zeros(shape=(1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] - min(rpr + 1, params['adc']) // 2 ** step), dtype=np.uint32)
                    conf_table[xb, wb, step, rpr]  = np.concatenate((conf, conf_pad), axis=-1)

                    values_pad = -1 * np.ones(shape=(                                              params['adc'] - min(rpr + 1, params['adc']) // 2 ** step), dtype=np.float32)
                    value_table[xb, wb, step, rpr] = np.concatenate((values, values_pad), axis=-1)

                    scale = 2**wb * 2**xb / q * ratio
                    error_table[xb][wb][step][rpr] = scale * mse
                    mean_table [xb][wb][step][rpr] = scale * mean
                    delay      [xb][wb][step][rpr] = row[xb][rpr]

                    if params['sar']:
                        sar = min(rpr + 1, params['adc'])
                        sar = 1 + np.ceil(np.log2(sar)) - step
                        delay[xb][wb][step][rpr] *= sar

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

    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    conf = np.zeros(shape=(8, 8, 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']), dtype=np.uint32)
    value = np.zeros(shape=(8, 8, 1 + params['adc']), dtype=np.float32)

    # TODO: need to verify this. no idea whats going on.
    # because we unrolled loop above, just index "valid" one at a time.
    if params['skip'] and params['cards']:
        rpr_range  = np.arange(1, params['max_rpr'] + 1).reshape(1, -1)
        step_range = 2 ** np.arange(params['max_step']).reshape(-1, 1)
        step_mask  = np.minimum(params['adc'], rpr_range) >= step_range
        valid = np.ones_like(error_table) * step_mask
        rpr_lut, step_lut = optimize_rpr(error_table, np.abs(mean_table), delay, valid, params['thresh'])

    # TODO
    # can do this in 1 go I think.
    # error = error_table[rpr_lut, step_lut]
    # 
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr = rpr_lut[xb][wb]
            step = step_lut[xb][wb] 
            error[xb][wb] = error_table[xb][wb][step][rpr-1]
            mean[xb][wb]  = mean_table[xb][wb][step][rpr-1]
            conf[xb][wb]  = conf_table[xb][wb][step][rpr-1]  # [WL, ON, ADC]
            value[xb][wb] = value_table[xb][wb][step][rpr-1] # [ADC]

    # step_lut = 2 ** step_lut
    # assert (np.sum(error) >= np.sum(np.abs(mean)))
    # print (rpr_lut)
    # print (step_lut)
    return rpr_lut, step_lut, conf, value
    
    
##########################################
    
    

