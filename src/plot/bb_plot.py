
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

results = np.load('../results.npy', allow_pickle=True)
# print (len(results))

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['rpr_alloc'])

####################

comp_pJ = 45e-15

# power plot is problem -> [0, 0], [1, 0] produce same result.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

for (skip, alloc, profile) in [(1, 'block', 1),  (1, 'layer', 1), (1, 'layer', 0), (0, 'layer', 1)]:
# for (skip, alloc, profile) in [(0, 'layer', 1)]:
# for (skip, alloc, profile) in [(1, 'block', 1)]:

    ######################################

    narrays = [10443, 1.5 * 2 ** 13, 2 ** 14, 1.5 * 2 ** 14, 2 ** 15]
    mac_per_cycles = []
    mac_per_pJs = []
    errors = []
    
    for narray in narrays:
        query = '(skip == %d) & (profile == %d) & (alloc == "%s") & (narray == %d)' % (skip, profile, alloc, narray)
        samples = df.query(query)
        mac_per_cycle = np.sum(samples['nmac']) / np.max(samples['cycle'])
        mac_per_cycles.append(mac_per_cycle)

        # cycles = np.array(samples['cycle'])
        # print (skip, alloc, profile, np.max(cycles), cycles)

    ######################################
        
    plt.plot(narrays, mac_per_cycles, marker='.')
    
    ######################################

plt.gcf().set_size_inches(9., 4.5)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.show()
            
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
