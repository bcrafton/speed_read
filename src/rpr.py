
import numpy as np
from scipy.stats import norm, binom

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

'''
we are thinking this will be a function of:
1) variance
2) adc
3) xbit
4) weight stats
5) quantization value
6) columns (we only do 256 columns at a time)
7) wbit (if we divide up wbit to different arrays)

break expected error into:
1) error from variance
2) error from rpr > adc
'''
def rpr(nrow, p, q, params):
    rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_lut[xb][wb] = params['adc']
        
    if not (params['skip'] and params['cards']):
        '''
        for key in sorted(rpr_lut.keys()):
            print (key, rpr_lut[key])
        print (np.average(list(rpr_lut.values())))
        '''
        return rpr_lut
    
    # counting cards:
    # ===============
    for wb in range(params['bpw']):
        for xb in range(params['bpa']):
            rpr_low = 1
            rpr_high = 16
            for rpr in range(rpr_low, rpr_high + 1):
                # we over approximate error in several ways:
                # 1) np.ceil(nrow / rpr)
                # 2) dont consider bias, relu
                # 3) we keep all 64 (x, w) below (1/64), bound to be far less than 1.
                scale = 2**wb * 2**xb
                mu, std = prob_err(p[wb], params['sigma'], params['adc'], rpr, np.ceil(nrow / rpr))
                # e = (scale / q) * 64 * std
                # e_mu = (scale / q) * 64 * mu
                e = (scale / q) * 5 * std
                e_mu = (scale / q) * 5 * mu

                if rpr == rpr_low:
                    rpr_lut[xb][wb] = rpr
                if (e < 1.) and (np.absolute(e_mu) < 1.):
                    rpr_lut[xb][wb] = rpr

    '''
    for key in sorted(rpr_lut.keys()):
        print (key, rpr_lut[key])
    print (np.average(list(rpr_lut.values())))
    '''
    return rpr_lut




