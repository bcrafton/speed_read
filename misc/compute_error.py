
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

rpr = 8
adc = 8
col_density = 0.5
sigma = 0.2

on = np.array(range(0, rpr + 1))

p = binom.pmf(on, rpr, col_density)

e_var = calc_e_var(rpr, sigma)
e_rpr = np.where(on > adc, on - adc, 0)

e = p * (e_var + e_rpr)

plt.plot(on, e)
plt.show()

#####################
