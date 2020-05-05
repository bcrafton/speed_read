
import numpy as np
from scipy.stats import norm

'''
print (norm.cdf([-3, -2, -1, 0, 1, 2, 3], loc=0, scale=1))

x = norm.ppf(q=[0.25, 0.5, 0.75], loc=0., scale=1.)
print (x)
'''

minval = norm.cdf(-3, loc=0, scale=1)
maxval = norm.cdf(3, loc=0, scale=1)
x = norm.ppf(q=np.linspace(minval, maxval, 10), loc=0., scale=1.)
print (x)
