
import numpy as np
from scipy.stats import norm, binom
    
##########################################

def expected_error(params, adc_count, row_count, rpr, nrow, bias):

    #######################
    # error from rpr <= adc
    #######################
    
    s  = np.arange(rpr + 1, dtype=np.float32)
    
    adc      = np.arange(params['adc'] + 1, dtype=np.float32).reshape(-1, 1)
    adc_low  = np.array([-1e6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
    adc_high = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)
    
    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-6)
    e = s - adc
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-6)

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

    weight = np.arange(65, dtype=np.float32)
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
    
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):

                #####################################################

                if rpr > params['adc']:
                    count = adc_count[xb, wb, rpr, sat_low:sat_high]
                    prob = count / (np.sum(count) + 1e-6)
                    weight = np.arange(sat_high - sat_low, dtype=np.float32)
                    bias = np.sum(prob * weight)
                    # print (xb, wb, rpr, np.around(prob, 3), bias)
                else:
                    bias = 0.

                #####################################################

                expected_cycles = np.ceil(nrow / params['wl']) * np.ceil(nrow_array[xb][rpr])

                scale = 2**wb * 2**xb
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], rpr=rpr, nrow=expected_cycles, bias=bias)
                scaled_mse = (scale / q) * 64. * mse
                scaled_mean = (scale / q) * 64. * mean

                # print (xb, wb, rpr, adc_count[xb, wb, rpr])

                if rpr == low:
                    rpr_lut[xb][wb] = rpr
                    bias_lut[xb][wb] = bias
                    mse_lut[xb][wb] = scaled_mse
                    mean_lut[xb][wb] = scaled_mean

                if scaled_mse < params['thresh']:
                    rpr_lut[xb][wb] = rpr
                    bias_lut[xb][wb] = bias
                    mse_lut[xb][wb] = scaled_mse
                    mean_lut[xb][wb] = scaled_mean

    # print (np.mean(mse_lut), np.mean(mean_lut))
    return rpr_lut, bias_lut
    
    
##########################################
    
    

