
import numpy as np

p = np.random.normal(loc=0.5, scale=0.01, size=32)

############################

def prob(ps):
    N = len(ps)
    dist = np.zeros(shape=(N + 1))
    dist[0] = 1.
    for p in ps:
        new_dist = np.copy(dist)
        new_dist[0:N]   = dist[0:N] * (1 - p)
        new_dist[1:N+1] += dist[0:N] * p
        dist = new_dist
        assert (np.sum(dist) >= 0.99)
        assert (np.sum(dist) <= 1.01)

    return dist

############################

print (prob(p))

############################
