
import math
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

var = 0.2
adc = 8
rpr = 10
row = 16
p = 0.8
samples = 100000

w = np.random.choice(a=[0, 1], size=[row, rpr, samples], replace=True, p=[1. - p, p])

y_true = np.sum(w, axis=(0, 1))

y = np.sum(w, axis=1)
y_var = np.random.normal(loc=0., scale=var * np.sqrt(y), size=np.shape(y))
y = y + y_var
y = np.around(y)
y = np.clip(y, 0, adc)
y = np.sum(y, axis=0)

print (np.shape(y), np.shape(y_true))

e = y - y_true
print (np.mean(e), np.std(e))
