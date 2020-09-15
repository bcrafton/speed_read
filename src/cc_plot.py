
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

# power plot is problem -> [0, 0], [1, 0] produce same result.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title('MSE')
ax2.set_title('Perf')
ax3.set_title('Perf/W')

for skip, cards, rpr_alloc in [(0, 0, 'dynamic'), (1, 0, 'dynamic'), (1, 1, 'static')]:

    ######################################

    sigmas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
    mac_per_cycles = []
    mac_per_pJs = []
    errors = []
    
    for sigma in sigmas:
        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f)' % (rpr_alloc, skip, cards, sigma)
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

    # plt.plot(sigmas, errors, marker='.', label=rpr_alloc)
    
    if cards:  label = 'cards'
    elif skip: label = 'skip'
    else:      label = 'baseline'
        
    ax1.plot(sigmas, errors,         label=label)
    ax2.plot(sigmas, mac_per_cycles, label=label)
    ax3.plot(sigmas, mac_per_pJs,    label=label)
        
    ######################################

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)

ax1.grid(True, linestyle='dotted')
ax2.grid(True, linestyle='dotted')
ax3.grid(True, linestyle='dotted')

fig.set_size_inches(13.5, 4.5)
plt.tight_layout()

plt.legend()
plt.ylim(bottom=0)
# plt.show()
plt.savefig('cc.png')
            
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
