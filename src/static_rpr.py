
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

'''
counts = self.profile[rpr][0:rpr+1]
values = np.array(range(rpr+1))

if rpr <= self.params['adc']:
    centroids = np.arange(0, self.params['adc'] + 1, step=1, dtype=np.float32)
else:
    centroids = kmeans(values=values, counts=counts, n_clusters=self.params['adc'] + 1)
    centroids = sorted(centroids)

p = counts / np.sum(counts)
s = values

mu, std = exp_err(s=s, p=p, var=self.params['sigma'], adc=centroids, rpr=rpr, row=np.ceil(p_avg * self.nrow / rpr))
'''

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

def expected_error(params, profile, rpr, nrow, bias):

    #######################
    # error from rpr <= adc
    #######################
    
    s  = np.arange(rpr + 1)
    
    adc      = np.arange(params['adc'] + 1).reshape(-1, 1)
    adc_low  = np.array([-1e6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
    adc_high = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)
    
    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-6)
    e = s - adc
    p = profile[rpr][0:rpr + 1] / np.sum(profile[rpr])
    
    exp_err = np.sum(p * pe * e)
    
    #######################
    # error from rpr > adc
    #######################
    
    e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - bias
    exp_err_bias = np.sum(p * pe * e)

    # print (rpr, exp_err, exp_err_bias)

    mu = np.sum(p * pe * e)
    std = np.sqrt(np.sum(p * pe * (e - mu) ** 2))

    mu = mu * nrow
    std = np.sqrt(std ** 2 * nrow)

    # if rpr == 16: assert (False)

    return mu, std

##########################################

def static_rpr(low, high, params, profile, nrow, q):
    
    ############
    
    # np.shape(profile) = (65, 65)
    # profile[8] is the adc counts for rpr=8
    # want -> (8, 8, 65, 65) = (xb, wb, rpr, count)
    
    ############
    
    # bias_lut can be 2 shapes:
    # [64], [8, 8]
    # 64 -> for each rpr value
    # [8, 8] -> the rpr values we care about
    # [8, 8] def better since we want to also do it by [xb, wb] later on 
    # EXCEPT that at the end of a row we dont turn on all 64.
    
    ############
    
    # below is close to correct, but not quite there yet.
    # problem is (+1, -1) thing with iterator and what not
    
    ############
    
    sat_low = params['adc']
    sat_high = high + 1
    
    bias_lut = np.zeros(shape=high + 1, dtype=np.float32)
    for rpr in range(sat_low, sat_high):
        count = profile[rpr][sat_low:sat_high]
        prob = count / np.sum(count)
        
        bias = 0.
        for i in range(sat_high - sat_low):
            bias = bias + prob[i] * i
        
        bias_lut[rpr] = bias

    ############

    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)

    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']
        
    if not (params['skip'] and params['cards']):
        return rpr_lut, bias_lut
    
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            for rpr in range(low, high + 1):
                scale = 2**wb * 2**xb
                # mu, std = prob_err(p[wb], params['sigma'], params['adc'], rpr, np.ceil(nrow / rpr))
                mu, std = expected_error(params=params, profile=profile, rpr=rpr, nrow=np.ceil(nrow / rpr), bias=bias_lut[rpr])

                e = (scale / q) * (64. / 2.) * std
                e_mu = (scale / q) * (64. / 2.) * mu

                if rpr == low:
                    rpr_lut[xb][wb] = rpr
                if (e < 1.) and (np.absolute(e_mu) < 1.):
                    rpr_lut[xb][wb] = rpr

    return rpr_lut, bias_lut
    
    
##########################################
    
    
