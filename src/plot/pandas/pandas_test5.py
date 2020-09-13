
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
# print (len(results))

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['rpr_alloc'])

####################

layer = 3
comp_pJ = 22. * 1e-12 / 32. / 16.

for cards in [0, 1]:
    for rpr_alloc in ['dynamic', 'centroids']:
    
        ######################################
    
        query = '(rpr_alloc == "%s") & (cards == %d) & (id == %d)' % (rpr_alloc, cards, layer)
        samples = df.query(query)
        if (len(samples) == 0):
            continue

        ######################################

        adc = np.stack(samples['adc'], axis=0)    
        energy = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_pJ, axis=1) 
        energy += samples['ron'] * 2e-16
        energy += samples['roff'] * 2e-16
        mac_per_pJ = samples['nmac'] / 1e12 / energy
        
        ######################################
        
        error = samples['std']
        
        ######################################
        
        mac_per_cycle = samples['nmac'] / samples['cycle']
        
        ######################################

        sigma = samples['sigma']
        plt.plot(sigma, mac_per_pJ, marker='.', label=rpr_alloc)
        
        ######################################

plt.legend()
plt.ylim(bottom=0)
plt.show()
            
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
