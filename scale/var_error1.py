
import math
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

##################
'''
on = 4
std = np.sqrt(on * 0.2 ** 2)

adc = 8
x = np.array(range(adc + 1))
a = np.clip(x - 0.5, 0, adc)
b = np.clip(x + 0.5, 0, adc) 

p = norm.cdf(x=b, loc=on, scale=std) - norm.cdf(x=a, loc=on, scale=std)
e = p * (x - on) ** 2
plt.plot(x, p)
plt.plot(x, e)
plt.show()
'''
##################


on = np.array(range(8 + 1)).reshape(1, -1)
std = np.sqrt(on * 0.2 ** 2)

x = np.array(range(8 + 1)).reshape(-1, 1)
on_a = np.clip(x - 0.5, 0, 8)
on_b = np.clip(x + 0.5, 0, 8)

##################

p = norm.cdf(x=on_b, loc=on, scale=std) - norm.cdf(x=on_a, loc=on, scale=std)
p = np.where(np.isnan(p), 0., p)
e = p * (x - on) ** 2

# print (e)

##################

# this looks wrong.
plt.plot(on.flatten(), p[:, 2].flatten())
plt.plot(on.flatten(), e[:, 2].flatten())
plt.show()

##################


