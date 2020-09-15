
import numpy as np
from scipy.stats import norm, binom

##########################################

def prob_err(p, var, adc, rpr, row):
    def prob_err_help(e, p, var, adc, rpr):
        psum = 0
        for s in range(1, rpr + 1):
            bin = binom.pmf(s, rpr, p)
            psum += ((s + e) < adc) * bin * (norm.cdf(e + 0.5, 0, var * np.sqrt(s)) - norm.cdf(e - 0.5, 0, var * np.sqrt(s)))
            psum += ((s + e) == adc) * bin * (1 - norm.cdf(adc - s - 0.5, 0, var * np.sqrt(s)))

        # zero case:
        psum += ((e - 0.5 < 0) * (0 < e + 0.5)) * binom.pmf(0, rpr, p)
        return psum
    
    s = np.array(range(-rpr, rpr+1))
    pe = prob_err_help(s, p, var, adc, rpr)
    mu = np.sum(pe * s)
    std = np.sqrt(np.sum(pe * (s - mu) ** 2))
    
    mu = mu * row
    std = np.sqrt(std ** 2 * row)
    return mu, std
    
##########################################

def exp_err(s, p, var, adc, rpr, row):
    assert (np.all(p <= 1.))
    assert (len(s) == len(p))

    adc = sorted(adc)
    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    pe = norm.cdf(adc_high, s, var * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, var * np.sqrt(s) + 1e-6)
    e = s - adc
    
    '''
    print (np.shape(pe))
    print (np.shape(e))
    print (np.shape(p))
    
    (9, 22)
    (9, 22)
    (22,)
    
    s =[1, 22]
    adc = [9, 1]
    '''
    
    # print (s.flatten())
    # print (adc.flatten())
    # print (e)
    # print (np.round(p * pe * e, 2))
    # print (adc_low.flatten())
    # print (adc_high.flatten())

    mu = np.sum(p * pe * e)
    std = np.sqrt(np.sum(p * pe * (e - mu) ** 2))

    mu = mu * row
    std = np.sqrt(std ** 2 * row)

    # print (rpr, (mu, std), adc.flatten())
    
    return mu, std
    
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
    nrow = np.sum(row_count * weight, axis=2) / (np.sum(row_count, axis=2) + 1e-6)
    
    # print (adc_count[0][0][16])
    # print (row_count[0][16])
    # print (nrow[0][16])
    # print (nrow[0])
    
    # it actually makes sense for this to be list of [1024]
    # they would all specify some # of rows they take.
    # 32 * 32 = 1024.
    # print (np.sum(row_count[0], axis=1))

    # assert (False)
    
    ############
    
    sat_low = params['adc']
    sat_high = high + 1
    
    '''
    bias_lut = np.zeros(shape=sat_high, dtype=np.float32)
    for rpr in range(sat_low, sat_high):
        count = np.sum(adc_count, axis=(0,1))[rpr][sat_low:sat_high]
        prob = count / np.sum(count)
        weight = np.arange(sat_high - sat_low, dtype=np.float32)
        bias = np.sum(prob * weight)
        bias_lut[rpr] = bias
    '''

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

                scale = 2**wb * 2**xb
                mse, mean = expected_error(params=params, adc_count=adc_count[xb][wb], row_count=row_count[xb], rpr=rpr, nrow=np.ceil(nrow[xb][rpr]), bias=bias)
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
    
    

