
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

results = np.load('results.npy', allow_pickle=True)
results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)
print (df.columns)

####################

block = df[ df['alloc'] == 'block' ]
layer = df[ df['alloc'] == 'layer' ]

####################

print (block['cycle'])
print (block['cycle'].to_numpy())
print (np.sum(block['cycle']))

####################
