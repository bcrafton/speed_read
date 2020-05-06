
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.stats import norm
import numpy.random as rand

#########################

psums = np.load('psums.npy', allow_pickle=True)
values, counts = np.unique(psums, return_counts=True)

# just use these, not this 300MB npy file.
print (values, counts)

#########################

# How far is each pt from the nearest centroid?
def distance(truth, test):
    truth = np.reshape(truth, (-1,1))
    test = np.reshape(test, (1,-1))
    return(np.min(np.absolute(np.subtract(truth,test)),axis=1))

# Some simple weighted error functions:    
def mean_sq_err(dist, freq):
    return(np.sum(np.square(dist) * freq))
    
def mean_abs_err(dist, freq):
    return(np.sum(np.absolute(dist) * freq))

# A "sparse" k-means implementation
def kmeans(values, counts, n_clusters=8,max_iter=10,n_init=50,err_func=mean_sq_err):
    
    # In case we need these:
    probs = counts/np.sum(counts)
    
    # k-means++ initialization:
    def k_means_pp():
        weighted_probs = probs
        clusters = np.zeros(n_clusters)
        for c in range(1, n_clusters):
            # 1: choose new cluster center using weighted chance
            clusters[c] = rand.choice(values, p=weighted_probs)
            # 2: compute new weights
            d = distance(values, clusters[0:c+1])
            weighted_probs = probs*np.square(d)
            weighted_probs = weighted_probs/np.sum(weighted_probs)
        return(clusters)
    
    # Iterate once thru the algorithm:
    def recompute_clusters(clusters):
        # Assign values
        v = np.reshape(values, (-1,1))
        c = np.reshape(clusters, (1,-1))
        d = np.absolute(v - c)
        # Turn this into a weighted selector matrix:
        # If a value is equal distance between N means,
        # each mean is adjusted by 1/N * frequency of value.
        s = 1.0*np.equal(0,d - np.min(d, axis=1).reshape((-1,1)))
        s = s * np.sum(s, axis=1).reshape((-1,1)) * probs.reshape((-1,1))
        s /= np.sum(s,axis=0).reshape((1,-1))
        
        # Now recompute cluster centers:
        cl = np.sum(s*values.reshape((-1,1)), axis=0)
        cl[0] = 0
        
        return(cl)                    
    
    min_err = 1e6
    min_cntrs = None
    for init in range(n_init):
        cl = k_means_pp()
        for it in range(max_iter):    
            cl = recompute_clusters(cl)
            mse = err_func(distance(values, cl), probs)
            if mse < min_err:
                min_err = mse
                min_cntrs = cl
    
    return(min_cntrs)         

min_cntrs = kmeans(values, counts, err_func=mean_sq_err)

probs = counts/np.sum(counts)
print(min_cntrs)
print(mean_sq_err(distance(values, min_cntrs), probs))
print(mean_sq_err(distance(values, np.asarray([0, 1, 3, 4, 5, 6, 7, 8])), probs))   