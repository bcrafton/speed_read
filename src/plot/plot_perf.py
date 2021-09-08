
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perf import perf

####################

def ld_to_dl(ld):
    dl = {}

    for i, d in enumerate(ld):
        for key in d.keys():
            value = d[key]
            if i == 0:
                dl[key] = [value]
            else:
                dl[key].append(value)

    return dl

####################

results = np.load('../results.npy', allow_pickle=True)

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['id'])
# print (df['narray'])

####################

# query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (example == %d) & (thresh == %f)' % (rpr_alloc, skip, cards, sigma, example, thresh)
# samples = df.query(query)

####################

counts = []
costs = []

layers = np.array(df['id'])
for layer in layers:
    query = '(id == %d)' % (layer)
    samples = df.query(query)
    #
    count = samples['count'][layer]
    counts.append(count)
    #
    nwl = samples['nwl'][layer]
    nbl = samples['nbl'][layer]
    cost = nwl * nbl
    costs.append(cost)

cycles = perf(counts=counts, costs=costs, resources=2 ** 10)
print (cycles)

####################















