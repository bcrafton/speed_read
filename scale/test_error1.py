import numpy as np
from scipy.stats import norm, binom
from numpy.random import binomial, normal
import matplotlib.pyplot as plt

# Fix parameters:
Nrpr = 10
Nadc = 8
std = 0.2
p = 0.8
number_reads = 16

reps = 1000
errors = np.empty(reps)

for rep in range(reps):
    weights = binomial(1,p,number_reads*Nrpr)
    
    out_true = np.sum(weights)
    
    devs = normal(0,std,number_reads*Nrpr)

    weights = weights*devs + weights

    weights = np.reshape(weights,(number_reads, Nrpr))

    y = np.sum(weights, axis=1)

    assert(y.shape == (number_reads,))

    y = np.around(y)
    
    y = np.minimum(y, Nadc)

    out_test = np.sum(y)
    
    errors[rep] = out_test - out_true

print(np.mean(errors), np.std(errors))
