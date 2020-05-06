
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm, binom

#########################

psums = np.load('psums.npy', allow_pickle=True)
values, counts = np.unique(psums, return_counts=True)

# just use these, not this 300MB npy file.
# print (values, counts)

#########################

kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=5, random_state=0)
kmeans.fit(values.reshape(-1, 1), counts)
centroids = np.round(kmeans.cluster_centers_[:, 0], 2)

#########################

# round centroids up or down ? 
# alright so do new clever error compute.

#########################

arg = np.argmin(np.absolute(psums - centroids), axis=1)
out = centroids[arg]

# print (sorted(centroids))
# print (np.unique(out, return_counts=True))

# plt.hist(out, bins=100)
# plt.show()

#########################
'''
# for every psum value
# figure out (%) it can end up in all adc states
# compute error for each of those states.

# how to figure out this (%) chance ? 
# cdf.

s = psums[0:10]
var = 0.1
centroids = sorted(centroids)

adc_low = np.copy(centroids)
adc_low[0] = -1e3
adc_high = np.append(centroids[1:], [1e3])
print (adc_low)
print (adc_high)

p = norm.cdf(adc_high, s, var * np.sqrt(s)) - norm.cdf(adc_low, s, var * np.sqrt(s))

print (s)
print (np.around(p, 3))
'''
#########################

# okay so new problem is clearly rounding.
# specifically looking at the case for 2.
# because we dont do the 0.5 thing right now
# but it really is as easy as picking: adc_low and adc_high.

# because then you just do the cdf
# and this clever error calc.

#########################

s = psums[0:10]
var = 0.1
centroids = sorted(centroids)

def adc_range(centroids):
    adc_low = np.zeros_like(centroids)
    adc_high = np.zeros_like(centroids)
    
    adc_low[0] = -1e2
    adc_high[-1] = 1e2
    
    for s in range(len(centroids) - 1):
        adc_high[s] = (centroids[s] + centroids[s + 1]) / 2
        adc_low[s + 1] = (centroids[s] + centroids[s + 1]) / 2

    return adc_low, adc_high

adc_low, adc_high = adc_range(centroids)

#########################

print (centroids)
print (adc_low)
print (adc_high)

p = norm.cdf(adc_high, s, var * np.sqrt(s)) - norm.cdf(adc_low, s, var * np.sqrt(s))

print (s)
print (np.around(p, 3))

#########################















