
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

comp_pJ = 22. * 1e-12 / 32. / 16.

for cards, rpr_alloc in ((0, 'dynamic'), (1, 'dynamic'), (1, 'centroids')):

    ######################################

    sigmas = [0.01, 0.10, 0.20, 0.30]
    mac_per_cycles = []
    mac_per_pJs = []
    errors = []
    
    for sigma in sigmas:
        query = '(rpr_alloc == "%s") & (cards == %d) & (sigma == %f)' % (rpr_alloc, cards, sigma)
        samples = df.query(query)
        
        mac_per_cycle = np.sum(samples['nmac']) / np.max(samples['cycle'])
        mac_per_cycles.append(mac_per_cycle)
        
        error = np.average(samples['std'])
        errors.append(error)

        adc = np.stack(samples['adc'], axis=0)    
        energy = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_pJ, axis=1) 
        energy += samples['ron'] * 2e-16
        energy += samples['roff'] * 2e-16
        mac_per_pJ = np.sum(samples['nmac']) / 1e12 / np.sum(energy)
        mac_per_pJs.append(mac_per_pJ)

    ######################################

    plt.plot(sigmas, mac_per_pJs, marker='.', label=rpr_alloc)
    
    ######################################

plt.legend()
plt.ylim(bottom=0)
plt.show()
            
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
