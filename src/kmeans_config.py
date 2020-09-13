
import math
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from scipy.stats import norm, binom
from kmeans import kmeans

#########################

def adc_range(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_low = np.zeros_like(adc, dtype=np.float32)
    adc_high = np.zeros_like(adc, dtype=np.float32)
    
    adc_low[0] = -1e2
    adc_high[-1] = 1e2
    
    for s in range(len(adc) - 1):
        adc_high[s] = (adc[s] + adc[s + 1]) / 2
        adc_low[s + 1] = (adc[s] + adc[s + 1]) / 2

    return adc_low, adc_high
    
#########################

def adc_floor(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_thresh = np.zeros_like(adc, dtype=np.float32)
    
    for s in range(len(adc) - 1):
        adc_thresh[s] = (adc[s] + adc[s + 1]) / 2

    adc_thresh[-1] = adc[-1]
    
    return adc_thresh

#########################

def exp_err(s, p, var, adc, rpr, row):
    assert (np.all(p <= 1.))
    assert (len(s) == len(p))

    adc = sorted(adc)
    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    pe = norm.cdf(adc_high, s, var * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, var * np.sqrt(s) + 1e-6)
    e = s - adc
    
    # print (s.flatten())
    # print (adc.flatten())
    # print (e)
    # print (np.round(p * pe * e, 2))
    # print (adc_low.flatten())
    # print (adc_high.flatten())

    mse = np.sqrt(np.sum((p * pe * e * row) ** 2))

    return mse

#########################

class KmeansConfig(Config):
    
    def rpr(self):
            
        adc_state = np.zeros(shape=(self.high + 1, self.params['adc'] + 1))
        adc_thresh = np.zeros(shape=(self.high + 1, self.params['adc'] + 1))
        
        rpr_dist = {}
        for rpr in range(self.low, self.high + 1):
            counts = self.profile[rpr][0:rpr+1]
            values = np.array(range(rpr+1))
            
            if rpr <= self.params['adc']:
                centroids = np.arange(0, self.params['adc'] + 1, step=1, dtype=np.float32)
            else:
                centroids = kmeans(values=values, counts=counts, n_clusters=self.params['adc'] + 1)
                centroids = sorted(centroids)
            
            p = counts / np.sum(counts)
            s = values

            p_avg = 1. 

            mse = exp_err(s=s, p=p, var=self.params['sigma'], adc=centroids, rpr=rpr, row=np.ceil(p_avg * self.nrow / rpr))
            rpr_dist[rpr] = {'mse': mse, 'centroids': centroids}
            
            adc_state[rpr] = 4 * np.array(centroids)
            adc_thresh[rpr] = adc_floor(centroids)
            
            if rpr == 1:
                adc_thresh[rpr][0] = 0.2
            
        rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                rpr_lut[xb][wb] = self.params['adc']
            
        if not (self.params['skip'] and self.params['cards']):
            return rpr_lut
        
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                for rpr in range(self.low, self.high + 1):
                
                    scale = 2**wb * 2**xb
                    mse = rpr_dist[rpr]['mse']
                    scaled_mse = (scale / self.q) * 64. * mse
                    
                    if rpr == self.low:
                        rpr_lut[xb][wb] = rpr
                    if scaled_mse < 1:
                        rpr_lut[xb][wb] = rpr

        return rpr_lut, adc_state, adc_thresh
        
#########################







        
        
        
