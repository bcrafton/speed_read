
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#########################

psums = np.load('psums.npy', allow_pickle=True)
values, counts = np.unique(psums, return_counts=True)

# just use these, not this 300MB npy file.
# print (values, counts)

#########################

kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=5, random_state=0)
kmeans.fit(values.reshape(-1,1), counts)
centroids = np.round(kmeans.cluster_centers_[:, 0], 2)
print (centroids)

#########################
'''
centroids = np.array( range(8) ) / 8 * np.max(values)

#########################

distance = np.sqrt((values.reshape(-1, 1) - centroids) ** 2)
binding = np.argmin(distance, axis=1)
print (binding)

centroids = values * counts
'''
