
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#########################

psums = np.load('psums.npy', allow_pickle=True)
values, counts = np.unique(psums, return_counts=True)

# just use these, not this 300MB npy file.
print (values, counts)

#########################
'''
points = np.array(range(8)) / 8 + (1 / 8 / 2)
centroids = np.quantile(psums, points, interpolation='nearest')
print (points, centroids)

plt.hist(psums)
plt.show()
'''
#########################

kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=2, n_init=1, random_state=0)
kmeans.fit(psums)
centroids = np.round(kmeans.cluster_centers_[:, 0], 2)
print (centroids)

#########################

def error(x, centroids)
    

#########################

