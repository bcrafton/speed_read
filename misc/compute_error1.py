
import math
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

def calc_e_var(adc, var):
    on = np.array(range(adc + 1)).reshape(1, -1)
    std = np.sqrt(on * var ** 2)

    x = np.array(range(adc + 1)).reshape(-1, 1)
    on_a = np.clip(x - 0.5, 0, adc)
    on_b = np.clip(x + 0.5, 0, adc)
    
    p = norm.cdf(x=on_b, loc=on, scale=std) - norm.cdf(x=on_a, loc=on, scale=std)
    p = np.where(np.isnan(p), 0., p)
    
    e = p * np.absolute(x - on)
    e = np.sum(e, axis=0)
    return e
    
#####################
    
def prob_err(e, p, var, adc, rpr):
    psum = 0
    # Use state variables Nrpr, p:
    for s in range(1, rpr + 1):
        bin = binom.pmf(s, rpr, p)
        c = s + e
        '''
        if (c < adc):  psum += bin * (norm.cdf((a + 0.5) / (std * np.sqrt(s))) - norm.cdf((a - 0.5) / (std*np.sqrt(s))))
        if (c == adc): psum += bin * (1 - norm.cdf((adc - s - 0.5) / (std * np.sqrt(s))))
        '''
        if (c < adc):  psum += bin * (norm.cdf(e + 0.5, 0, var * np.sqrt(s)) - norm.cdf(e - 0.5, 0, var * np.sqrt(s)))
        if (c == adc): psum += bin * (1 - norm.cdf(adc - s - 0.5, 0, var * np.sqrt(s)))

    # The PDF for normal becomes dirac delta when s = 0, so prob. over some range is 1 as long as range includes 0:
    if (e - 0.5 < 0 < e + 0.5): psum += binom.pmf(0, rpr, p)

    return psum
    
#####################

rpr = 10
adc = 8
col_density = 0.8
sigma = 0.2

es = []
for s in range(-4, 4): 
    e = prob_err(s, col_density, sigma, adc, rpr)
    es.append(e)

plt.plot(range(-4, 4), es)
plt.show()

#####################







