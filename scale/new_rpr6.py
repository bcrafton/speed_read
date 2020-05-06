
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

def adc_range(centroids):
    adc_low = np.zeros_like(centroids)
    adc_high = np.zeros_like(centroids)
    
    adc_low[0] = -1e2
    adc_high[-1] = 1e2
    
    for s in range(len(centroids) - 1):
        adc_high[s] = (centroids[s] + centroids[s + 1]) / 2
        adc_low[s + 1] = (centroids[s] + centroids[s + 1]) / 2

    return adc_low, adc_high

#########################

def exp_err(s, p, var, adc, rpr, row):
    assert (np.all(p <= 1.))
    assert (len(s) == len(p))

    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    pe = norm.cdf(adc_high, s, var * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, var * np.sqrt(s) + 1e-6)
    e = s - adc

    mu = np.sum(p * pe * e)
    std = np.sqrt(np.sum(p * pe * (e - mu) ** 2))

    mu = mu * row
    std = np.sqrt(std ** 2 * row)
    return mu, std

#########################

s = values
p = counts / np.cumsum(counts)
var = 0.1
adc = sorted(centroids)
rpr = 12
row = 1

#########################

mu, std = exp_err(s=s, p=p, var=var, adc=adc, rpr=rpr, row=row)
print (mu, std)

#########################















