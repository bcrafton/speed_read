
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

def expected_error(params, step, adc_hist, row, adc_thresh, adc_value):

    # pe

    adc      =          adc_value.T.reshape(1 + params['adc'] // 2 ** step, params['max_rpr'], 1, 1)
    adc_low  = adc_thresh[:, :-1].T.reshape(1 + params['adc'] // 2 ** step, params['max_rpr'], 1, 1)
    adc_high = adc_thresh[:,  1:].T.reshape(1 + params['adc'] // 2 ** step, params['max_rpr'], 1, 1)

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

    # row
    row = np.reshape(row, (params['max_rpr'], 1, 1))

    ########################################################################

    # p
    p = adc_hist.astype(np.float32)
    p = p / np.sum(p, axis=(1, 2), keepdims=True)

    ########################################################################

    # e
    s = np.arange(0, params['max_rpr']+1, dtype=np.float32)
    e = adc - s
    
    ########################################################################

    assert np.allclose(np.sum(p,      axis=(1, 2)),    1)
    assert np.allclose(np.sum(pe,     axis=0),         1)
    assert np.allclose(np.sum(pe * p, axis=(0, 2, 3)), 1)

    # [xb, wb], [adc, rpr, hrs, lrs]

    '''
    print (np.shape(p))
    print (np.shape(pe))
    print (np.shape(e))
    print (np.shape(row))
    assert (False)
    '''

    error = p * pe * e * row
    mse  = np.sum(np.absolute(error), axis=(0, 2, 3))
    mean = np.sum(           (error), axis=(0, 2, 3))

    return mse, mean

##########################################

def static_rpr(id, params, q):
    assert (q > 0)

    ############

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']
    step_lut = np.zeros(shape=(8, 8), dtype=np.int32)

    ############
    '''
    # cannot return early -- need {conf, value}
    if not (params['skip'] and params['cards']):
        step_lut = 2 ** step_lut
        return rpr_lut, step_lut, 0, 0
    '''
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

    # print (adc[:, :, :, 0])
    # this should all be 0.
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for step in range(params['max_step']):

                # {step, sar}
                # sar -> dosnt matter just performance related
                # step -> need to pad {conf, value}

                # thresholds:
                # tell it to pick (adc // 2 ** step) thresholds. 
                # dont pad here -> cdf will get fucked in confusion

                # conf:
                # need to pad {conf, value} after cdf

                thresh, values = thresholds(adc[xb, wb, 1:], params['adc'] // 2 ** step, method='kmeans')
                conf = confusion(thresh, params['max_rpr'], params['adc'] // 2 ** step, params['hrs'], params['lrs'])

                mse, mean = expected_error(params, step, adc[xb, wb, 1:], row[xb], thresh, values)
                assert np.all(mse >= np.abs(mean))

                conf_pad =       np.zeros(shape=(params['max_rpr'], 1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] - params['adc'] // 2 ** step), dtype=np.uint32)
                conf_table[xb, wb, step]  = np.concatenate((conf, conf_pad), axis=-1)

                values_pad = -1 * np.ones(shape=(params['max_rpr'],                                               params['adc'] - params['adc'] // 2 ** step), dtype=np.float32)
                value_table[xb, wb, step] = np.concatenate((values, values_pad), axis=-1)

                # assert (False)
                # so we just padded.
                # no we need to send it through expected error
                # really its just figuring out how to make step work here and we are good.

                scale = 2**wb * 2**xb / q * ratio
                error_table[xb][wb][step] = scale * mse
                mean_table[xb][wb][step] = scale * mean
                delay[xb][wb][step] = row[xb]

                if params['sar']:
                    sar = np.arange(1, params['max_rpr'] + 1)
                    sar = np.minimum(sar, params['adc'])
                    sar = 1 + np.ceil(np.log2(sar)) - step
                    delay[xb][wb][step] *= sar


    '''
    print (delay[0, 0, 0, :])
    print (delay[0, 0, 1, :])
    print (delay[0, 0, 2, :])
    '''

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
    if params['skip'] and params['cards']:
        rpr_range  = np.arange(1, params['max_rpr'] + 1).reshape(1, -1)
        step_range = 2 ** np.arange(params['max_step']).reshape(-1, 1)
        step_mask  = np.minimum(params['adc'], rpr_range) >= step_range
        valid = np.ones_like(error_table) * step_mask
        rpr_lut, step_lut = optimize_rpr(error_table, np.abs(mean_table), delay, valid, params['thresh'])

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
    
    

