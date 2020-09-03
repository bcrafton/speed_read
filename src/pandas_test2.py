
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

# example:
# y_mean[skip][cards][alloc][profile][rpr_alloc][layer]

'''
block = df[ df['alloc'] == 'block' ][ df['rpr_alloc'] == 'centroids' ]
print (block)

block = df.query('(alloc == "block") & (rpr_alloc == "centroids")')
print (block)
'''
####################

x = df.query('(alloc == "block") & (rpr_alloc == "centroids") & (profile == 1)')
mac_per_cycle = x['nmac'] / x['cycle']
print (mac_per_cycle)

####################

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
