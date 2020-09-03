
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
print (len(results))

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df)

####################

for alloc in ['block', 'layer']:
    query = '(alloc == "%s") & (id == 5)' % (alloc)
    samples = df.query(query)
    mac_per_cycle = samples['nmac'] / samples['cycle']
    sigma = samples['sigma']
    plt.scatter(sigma, mac_per_cycle)

plt.show()
            
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
