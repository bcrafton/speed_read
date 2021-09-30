
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

def compute_error(val, pmf):
    mean = np.sum(val * pmf)
    mse = np.sqrt(np.sum(pmf * (val - mean) ** 2))
    mae = np.sum(pmf * np.abs(val - mean))
    return mse, mae, mean

##########################################

eps = 1e-20
inf = 1e20

def expected_error(params, states, rpr, profile, row, row_avg, adc_thresh, adc_value):

    # pe
    '''
    adc_center =       adc_value.reshape(states, 1, 1)
    adc_low    = adc_thresh[:-1].reshape(states, 1, 1)
    adc_high   = adc_thresh[ 1:].reshape(states, 1, 1)

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
    '''

    ########################################################################

    adc_center = adc_value.reshape(states, 1, 1)

    pe = confusion(THRESH=adc_thresh, RPR=params['max_rpr'], ADC=states, HRS=params['hrs'], LRS=params['lrs'])
    # print (np.shape(pe))
    pe = np.transpose(pe, (2, 0, 1))
    pe = pe / np.sum(pe, axis=0, keepdims=True)

    ########################################################################

    # p
    p = profile.astype(np.float64)
    p = p / np.sum(p)

    ########################################################################

    # e
    s = np.arange(0, params['max_rpr'] + 1, dtype=np.float32)
    e = adc_center - s

    ########################################################################

    assert np.allclose(np.sum(p             ), 1)
    assert np.allclose(np.sum(pe,     axis=0), 1)
    assert np.allclose(np.sum(pe * p        ), 1)
 
    error = p * pe * e
    mse_ref  = np.sum(np.absolute(error)) * row_avg
    mean_ref = np.sum(           (error)) * row_avg
 
    p = profile.astype(np.float64)
    p = p / (np.sum(p, axis=(0, 1)) + 1e-6)
    pe_mask = np.abs(e) > 0
    return mse_ref, mean_ref, np.sum(pe * pe_mask * p)

    ########################################################################
    '''
    ps = []
    es = []

    (wls, ons) = np.where(p > 0)
    for wl, on in zip(wls, ons):
        [pes] = np.where(pe[:, wl, on] > 0)
        pes = np.array(pes)
        _p = p[wl, on] * pe[:, wl, on][pes]
        _e = adc_value[pes] - on
        ps.extend( _p.tolist() )
        es.extend( _e.tolist() )

    ps = np.array(ps)
    es = np.array(es)
    _, mae, mean = compute_error(es, ps)

    mse = mae * row_avg
    mean = mean * row_avg
    '''
    ########################################################################

    # print ('mae', mse, mse_ref)
    # print ('mean', mean, mean_ref)

    ########################################################################

    return mse, mean

##########################################

def static_rpr(id, params, q):
    assert (q > 0)

    rpr_lut = np.ones(shape=(8, 8), dtype=np.int32) * params['adc']

    # RPR = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]) - 1
    # ADC = np.array([1]) - 1
    # SAR = np.array([0, 2, 3, 4, 5, 6])

    delay_table  = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    energy_table = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    error_table  = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    mean_table   = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    p_table      = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    valid_table  = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))
    area_table   = np.zeros(shape=(8, 8, len(params['rprs']), len(params['adcs']), len(params['sars'])))

    thresh_table = {}
    value_table = {}

    profile = np.load('./profile/%d.npy' % (id), allow_pickle=True).item()
    assert (params['wl'] == profile['wl'])
    assert (params['bl'] == profile['bl'])
    assert (params['max_rpr'] == profile['max_rpr'])
    row = profile['row']
    row_avg = np.maximum(1, profile['row_avg'])
    ratio = profile['ratio']
    profile = profile['adc']
    params['ratio'] = ratio

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr_idx, rpr in enumerate(params['rprs']):
                for adc_idx, adc in enumerate(params['adcs']):
                    for sar_idx, sar in enumerate(params['sars']):
                        states = (adc + 1) ** sar
                        '''
                        if rpr+1 < states: continue
                        '''
                        thresh, values = thresholds(counts=profile[xb, wb, rpr_idx],
                                                    adc=adc,
                                                    sar=sar,
                                                    method=params['method'])

                        thresh_table[(xb, wb, rpr_idx, adc_idx, sar_idx)] = thresh
                        value_table[(xb, wb, rpr_idx, adc_idx, sar_idx)] = values
                        assert (len(thresh) == states + 1)
                        assert (len(values) == states + 0)

                        mse, mean, p = expected_error(params=params,
                                                      states=states,
                                                      rpr=rpr,
                                                      profile=profile[xb, wb, rpr_idx],
                                                      row=row[xb][rpr - 1],
                                                      row_avg=row_avg[xb][rpr - 1],
                                                      adc_thresh=thresh, 
                                                      adc_value=values)

                        # assert np.all(mse >= np.abs(mean))
                        scale = 2**wb * 2**xb / q * ratio
                        error_table[xb][wb][rpr_idx][adc_idx][sar_idx] = scale * mse
                        mean_table [xb][wb][rpr_idx][adc_idx][sar_idx] = scale * mean
                        p_table    [xb][wb][rpr_idx][adc_idx][sar_idx] = p
                        valid_table[xb][wb][rpr_idx][adc_idx][sar_idx] = 1
                        delay_table[xb][wb][rpr_idx][adc_idx][sar_idx] = row_avg[xb][rpr - 1] * sar
                        adc_energy = (sar * adc * params['adc_energy'])
                        sar_energy = (sar *       params['sar_energy'])
                        energy_table[xb][wb][rpr_idx][adc_idx][sar_idx] = row_avg[xb][rpr - 1] * (sar_energy + adc_energy)
                        # not currently used:
                        area_table[xb][wb][rpr_idx][adc_idx][sar_idx] = adc + (sar - 1)

    assert (np.sum(valid_table[:, :, 0, 0, 0]) == 64)
    assert (np.sum(mean_table[:, :, 0, 0, 0]) >= -params['thresh'])
    assert (np.sum(mean_table[:, :, 0, 0, 0]) <=  params['thresh'])
    assert (np.sum(np.min(error_table, axis=(2, 3, 4))) <= params['thresh'])

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
    error_table = round_fraction(error_table, 1e-4) # - round_fraction(np.absolute(mean_table), 1e-4)
    mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)
    # 
    # error_table = round_fraction(error_table, 1e-4)
    # mean_table = round_fraction(mean_table, 1e-4)

    if params['skip'] and params['cards']:
        if   params['opt'] == 'delay':  minimize = delay_table
        elif params['opt'] == 'energy': minimize = energy_table
        else:                           assert (False)
        rpr_lut, adc_lut, sar_lut, N = optimize_rpr(error=error_table, 
                                                    mean=mean_table, 
                                                    delay=minimize,
                                                    valid=valid_table, 
                                                    area_adc=params['adc_area'],
                                                    area_sar=params['sar_area'],
                                                    area=params['area'],
                                                    threshold=params['thresh'],
                                                    Ns=params['Ns'])
    else:
        # A: np.argmax(params['rprs'])
        # B: (len(params['rprs']) - 1)
        rpr_lut = np.argmax(params['rprs']) * np.ones(shape=(params['bpa'], params['bpw']), dtype=int)
        adc_lut = np.argmax(params['adcs']) * np.ones(shape=(params['bpa'], params['bpw']), dtype=int)
        sar_lut = np.argmax(params['sars']) * np.ones(shape=(params['bpa'], params['bpw']), dtype=int)
        N = 1

    mean  = np.zeros(shape=(8, 8))
    error = np.zeros(shape=(8, 8))
    p_lut = np.zeros(shape=(8, 8))
    row   = np.zeros(shape=(8, 8))
    conf  = np.zeros(shape=(8, 8, 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']), dtype=np.uint32)
    value = np.zeros(shape=(8, 8, 1 + params['adc']), dtype=np.float32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            scale = 2**wb * 2**xb / q * ratio
            ##########################################
            rpr_idx  = rpr_lut[xb][wb]
            adc_idx  = adc_lut[xb][wb]
            sar_idx  = sar_lut[xb][wb]

            rpr  = params['rprs'][rpr_idx]
            adc  = params['adcs'][adc_idx]
            sar  = params['sars'][sar_idx]

            rpr_lut[xb][wb] = rpr
            adc_lut[xb][wb] = adc
            sar_lut[xb][wb] = sar
            ##########################################
            states = (adc + 1) ** sar

            error[xb][wb] = error_table[xb][wb][rpr_idx][adc_idx][sar_idx] / scale
            mean[xb][wb]  = mean_table[xb][wb][rpr_idx][adc_idx][sar_idx] / scale
            p_lut[xb][wb] = p_table[xb][wb][rpr_idx][adc_idx][sar_idx]
            row[xb][wb]   = row_avg[xb][rpr - 1]

            conf1 = confusion(THRESH=thresh_table[(xb, wb, rpr_idx, adc_idx, sar_idx)], 
                              RPR=params['max_rpr'], 
                              ADC=states, 
                              HRS=params['hrs'], 
                              LRS=params['lrs'])
            conf2 = np.zeros(shape=(1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] + 1 - states), dtype=np.uint64)
            conf[xb][wb] = np.concatenate((conf1, conf2), axis=-1)

            values1 = value_table[(xb, wb, rpr_idx, adc_idx, sar_idx)]
            values2 = -1 * np.ones(shape=(params['adc'] + 1 - states), dtype=np.float32)
            value[xb][wb] = np.concatenate((values1, values2), axis=-1)
            ##########################################

    print (rpr_lut)
    # print (adc_lut)
    print (sar_lut)
    # print (error)
    print (np.around(row, 3))
    print (value[0][0])
    print (value[0][2])

    return rpr_lut, adc_lut, sar_lut, N, conf, value, error, mean, p_lut

##########################################
    
    

