
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
    adc_center =       adc_value.reshape(1 + adc, 1, 1)
    adc_low    = adc_thresh[:-1].reshape(1 + adc, 1, 1)
    adc_high   = adc_thresh[ 1:].reshape(1 + adc, 1, 1)

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

    if params['method'] == 'kmeans': STEP = np.array([1]) - 1
    else:                            STEP = np.array([1, 2, 4, 8, 16]) - 1
    RPR = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]) - 1
    ADC = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]) - 1
    SAR = np.array([0, 2, 3, 4, 5, 6])

    # STEP = np.array([1, 2, 4]) - 1
    # RPR  = np.array([1, 2, 4, 8, 16]) - 1
    # ADC  = np.array([1, 2, 4, 8, 16]) - 1

    ############

    delay_table = np.zeros(shape=(8, 8, len(STEP), len(RPR), len(ADC), len(SAR)))
    error_table = np.zeros(shape=(8, 8, len(STEP), len(RPR), len(ADC), len(SAR)))
    mean_table  = np.zeros(shape=(8, 8, len(STEP), len(RPR), len(ADC), len(SAR)))
    valid_table = np.zeros(shape=(8, 8, len(STEP), len(RPR), len(ADC), len(SAR)))
    area_table  = np.zeros(shape=(8, 8, len(STEP), len(RPR), len(ADC), len(SAR)))

    thresh_table = {}
    value_table = {}

    profile = np.load('./profile/%d.npy' % (id), allow_pickle=True).item()
    row = np.maximum(1, profile['row'])
    ratio = profile['ratio']
    profile = profile['adc']

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for step_idx, step in enumerate(STEP):
                for rpr_idx, rpr in enumerate(RPR):
                    for adc_idx, adc in enumerate(ADC):
                        for sar_idx, sar in enumerate(SAR):
                            states = (adc + 1) * (2 ** sar)
                            if (rpr + 1) < states: continue

                            thresh, values = thresholds(counts=profile[xb, wb, rpr + 1],
                                                        step=step+1,
                                                        adc=states,
                                                        method=params['method'])

                            thresh_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)] = thresh
                            value_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)] = values
                            assert (len(thresh) == states + 2)
                            assert (len(values) == states + 1)

                            mse, mean = expected_error(params=params, 
                                                       step=step+1, 
                                                       rpr=rpr+1,
                                                       adc=states,
                                                       profile=profile[xb, wb, rpr + 1], 
                                                       row=row[xb][rpr], 
                                                       adc_thresh=thresh, 
                                                       adc_value=values)

                            assert np.all(mse >= np.abs(mean))
                            scale = 2**wb * 2**xb / q * ratio
                            error_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx] = scale * mse
                            mean_table [xb][wb][step_idx][rpr_idx][adc_idx][sar_idx] = scale * mean
                            valid_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx] = 1
                            # not accurate:
                            delay_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx] = row[xb][rpr] * (sar + 1)
                            # not currently used:
                            area_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx] = (adc + 1) + 1*(sar > 0)

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
        step_lut, rpr_lut, adc_lut, sar_lut, N = optimize_rpr(error=error_table, 
                                                              mean=np.abs(mean_table), 
                                                              delay=delay_table, 
                                                              valid=valid_table, 
                                                              area_adc=np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]),
                                                              area_sar=np.array([0, 1, 1.25, 1.50, 1.75, 2.00]),
                                                              area=params['area'],
                                                              threshold=params['thresh'])

    mean = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    conf = np.zeros(shape=(8, 8, 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']), dtype=np.uint32)
    value = np.zeros(shape=(8, 8, 1 + params['adc']), dtype=np.float32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            ##########################################
            step_idx = step_lut[xb][wb]
            rpr_idx  = rpr_lut[xb][wb]
            adc_idx  = adc_lut[xb][wb]
            sar_idx  = sar_lut[xb][wb]

            step = STEP[step_idx]
            rpr  = RPR[rpr_idx]
            adc  = ADC[adc_idx]
            sar  = SAR[sar_idx]

            step_lut[xb][wb] = step + 1
            rpr_lut[xb][wb] = rpr + 1
            adc_lut[xb][wb] = adc + 1
            sar_lut[xb][wb] = sar
            ##########################################
            states = (adc + 1) * (2 ** sar)

            error[xb][wb] = error_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx]
            mean[xb][wb]  = mean_table[xb][wb][step_idx][rpr_idx][adc_idx][sar_idx]

            conf1 = confusion(THRESH=thresh_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)], 
                              RPR=params['max_rpr'], 
                              ADC=states, 
                              HRS=params['hrs'], 
                              LRS=params['lrs'])
            conf2 = np.zeros(shape=(1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] - states), dtype=np.uint32)
            conf[xb][wb] = np.concatenate((conf1, conf2), axis=-1)

            values1 = value_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)]
            values2 = -1 * np.ones(shape=(params['adc'] - states), dtype=np.float32)
            value[xb][wb] = np.concatenate((values1, values2), axis=-1)
            ##########################################

    return step_lut, rpr_lut, adc_lut, sar_lut, N, conf, value

##########################################
    
    

