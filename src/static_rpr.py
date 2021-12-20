
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
    mse = np.sum(pmf * (val ** 2))
    mae = np.sum(pmf * np.abs(val - mean))
    return mse, mae, mean

##########################################

eps = 1e-20
inf = 1e20

def expected_error(params, states, xb, wb, rpr, sar, profile, scale, row, row_avg, adc_thresh, adc_value):

    ########################################################################

    pe = confusion(THRESH=adc_thresh, RPR=params['max_rpr'], ADC=states, HRS=params['hrs'], LRS=params['lrs'])
    pe = pe / np.sum(pe, axis=-1, keepdims=True)

    ########################################################################

    # p
    p = profile.astype(np.float64)
    p = p / np.sum(p)
    p = p.reshape(params['max_rpr'] + 1, params['max_rpr'] + 1, 1)

    ########################################################################

    # e
    s = np.arange(params['max_rpr'] + 1, dtype=np.float64).reshape(-1, 1)
    e = adc_value - s

    ########################################################################

    assert np.allclose(np.sum(p             ), 1)
    assert np.allclose(np.sum(pe,     axis=2), 1)
    assert np.allclose(np.sum(pe * p        ), 1)

    ########################################################################

    error = e * scale

    ########################################################################

    mae  = np.sum(p * pe * np.abs(error))
    mae  = mae * row_avg

    ########################################################################

    mean = np.sum(p * pe * error)
    mse  = np.sqrt(np.sum(p * pe * error ** 2))
    std  = np.sqrt(mse ** 2 - mean ** 2)

    mean = mean * row_avg
    # 
    # std = std * np.sqrt(row_avg)
    # 
    rows = np.arange(len(row))
    std = np.sqrt(np.sum(std ** 2 * row * rows))
    # 
    mse  = (mean ** 2) + (std ** 2)

    ########################################################################

    return mse, mean, np.sum(p * pe * (np.abs(e) > 0))

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
    # row_avg = np.maximum(1, profile['row_avg'])
    # assert (np.all(profile['row_avg'] >= 1))
    row_avg = profile['row_avg']
    ratio = profile['ratio']
    profile = profile['adc']
    params['ratio'] = ratio

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr_idx, rpr in enumerate(params['rprs']):
                for adc_idx, adc in enumerate(params['adcs']):
                    for sar_idx, sar in enumerate(params['sars']):
                        states = (adc + 1) ** sar
                        if rpr + 1 < states: continue

                        if sar > 1: method = params['method']
                        else:       method = 'normal'
                        thresh, values = thresholds(counts=profile[xb, wb, rpr_idx],
                                                    adc=adc,
                                                    sar=sar,
                                                    method=method)

                        thresh_table[(xb, wb, rpr_idx, adc_idx, sar_idx)] = thresh
                        value_table[(xb, wb, rpr_idx, adc_idx, sar_idx)] = values
                        assert (len(thresh) == states + 1)
                        assert (len(values) == states + 0)

                        sign_xb = -1 if (xb == 7) else 1
                        sign_wb = -1 if (wb == 7) else 1
                        scale = sign_xb * sign_wb * 2**wb * 2**xb / q
                        mse, mean, pe = expected_error(params=params,
                                                       states=states,
                                                       xb=xb,
                                                       wb=wb,
                                                       rpr=rpr,
                                                       sar=sar,
                                                       profile=profile[xb, wb, rpr_idx],
                                                       scale=scale,
                                                       row=row[xb][rpr - 1],
                                                       row_avg=max(1, row_avg[xb][rpr - 1]),
                                                       adc_thresh=thresh, 
                                                       adc_value=values)

                        '''
                        if (xb == 0) and (wb == 2) and (rpr_idx == 6) and (sar_idx == 3):
                            print ('p value!', p)
                        '''

                        error_table[xb][wb][rpr_idx][adc_idx][sar_idx] = mse
                        mean_table [xb][wb][rpr_idx][adc_idx][sar_idx] = mean
                        p_table    [xb][wb][rpr_idx][adc_idx][sar_idx] = pe
                        valid_table[xb][wb][rpr_idx][adc_idx][sar_idx] = 1 # (pe <= params['pe'])
                        delay_table[xb][wb][rpr_idx][adc_idx][sar_idx] = row_avg[xb][rpr - 1] * sar
                        adc_energy = (sar * adc * params['adc_energy'])
                        sar_energy = (sar *       params['sar_energy'])
                        energy_table[xb][wb][rpr_idx][adc_idx][sar_idx] = row_avg[xb][rpr - 1] * (sar_energy + adc_energy)
                        # not currently used:
                        area_table[xb][wb][rpr_idx][adc_idx][sar_idx] = adc + (sar - 1)

    ###########################################################

    error_table = round_fraction(error_table, 1e-4)
    mean_table = np.sign(mean_table) * round_fraction(np.absolute(mean_table), 1e-4)

    ###########################################################
    '''
    where = np.where(valid_table == False)
    error_table[where] = 1e9
    mean_table[where] = 1e9

    min_mse = np.sqrt(np.sum( np.min(mean_table ** 2 + error_table, axis=(2, 3, 4)) ))
    max_mse = np.sqrt(np.sum( np.max(mean_table ** 2 + error_table, axis=(2, 3, 4)) ))
    print ('min_mse', min_mse, 'max_mse', max_mse)
    assert min_mse <= params['thresh']

    min_mse = np.sqrt(np.sum( (mean_table[:,:,0,0,0] ** 2 + error_table[:,:,0,0,0]) ))
    print ('min_mse', min_mse)
    assert min_mse <= params['thresh']

    assert np.all( valid_table[:,:,0,0,0] == 1 )
    '''
    ###########################################################

    if params['skip'] and params['cards']:
        if   params['opt'] == 'delay':  minimize = round_fraction(delay_table, 4)
        elif params['opt'] == 'energy': minimize = round_fraction(energy_table, 4)
        else:                           assert (False)
        rpr_lut, adc_lut, sar_lut, N = optimize_rpr(error=error_table, 
                                                    mean=mean_table, 
                                                    delay=minimize,
                                                    valid=valid_table, 
                                                    prob=p_table,
                                                    area_adc=params['adc_area'],
                                                    area_sar=params['sar_area'],
                                                    area=params['area'],
                                                    scale=1,
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
    conf  = np.zeros(shape=(8, 8, 1 + params['max_rpr'], 1 + params['max_rpr'], 1 + params['adc']), dtype=np.uint64)
    value = np.zeros(shape=(8, 8, 1 + params['adc']), dtype=np.float32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
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

            error[xb][wb] = error_table[xb][wb][rpr_idx][adc_idx][sar_idx]
            mean[xb][wb]  = mean_table[xb][wb][rpr_idx][adc_idx][sar_idx]
            p_lut[xb][wb] = p_table[xb][wb][rpr_idx][adc_idx][sar_idx]
            row[xb][wb]   = row_avg[xb][rpr - 1]

            '''
            if (xb == 0) and (wb == 2):
                print ()
                print (states, thresh_table[(xb, wb, rpr_idx, adc_idx, sar_idx)])
                print ()
            '''

            conf1 = confusion(THRESH=thresh_table[(xb, wb, rpr_idx, adc_idx, sar_idx)], 
                              RPR=params['max_rpr'], 
                              ADC=states, 
                              HRS=params['hrs'], 
                              LRS=params['lrs'])
            #print (conf1.dtype)
            #print (np.max(conf1))
            conf2 = np.zeros(shape=(1 + params['max_rpr'], 1 + params['max_rpr'], params['adc'] + 1 - states), dtype=np.uint64)
            conf[xb][wb] = np.concatenate((conf1, conf2), axis=-1)
            #print (conf.dtype)
            #print (np.max(conf))

            values1 = value_table[(xb, wb, rpr_idx, adc_idx, sar_idx)]
            values2 = -1 * np.ones(shape=(params['adc'] + 1 - states), dtype=np.float32)
            value[xb][wb] = np.concatenate((values1, values2), axis=-1)
            ##########################################

    # print (rpr_lut)
    # print (sar_lut)

    return rpr_lut, adc_lut, sar_lut, N, conf, value, error, mean, p_lut

##########################################
    
    

