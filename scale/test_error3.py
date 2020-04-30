
import math
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

var = 0.2
adc = 8
rpr = 16
p = 0.8
samples = 100000

w = np.random.choice(a=[0, 1], size=[rpr, samples], replace=True, p=[1. - p, p])

y_true = np.sum(w, axis=0)

y = np.sum(w, axis=0)
y_var = np.random.normal(loc=0., scale=var * np.sqrt(y), size=np.shape(y))
y = y + y_var
y = np.around(y)
y = np.clip(y, 0, adc)
y = np.sum(y, axis=0)

# print (np.shape(y), np.shape(y_true))

# e = y - y_true
# print (np.mean(e), np.std(e))

######################

from sklearn.cluster import KMeans

print (np.min(y_true))
print (np.max(y_true))

x = range(adc, rpr + 1)
y = np.reshape(y_true, (-1, 1))

wcss = []

for i in x:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(y)
    wcss.append(kmeans.inertia_)
    
plt.plot(x, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
