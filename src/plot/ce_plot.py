
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

comp_pJ = 22. * 1e-12 / 32. / 16.

# power plot is problem -> [0, 0], [1, 0] produce same result.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

color = {
2 ** 12:       'red',
1.5 * 2 ** 12: 'orange',
2 ** 13:       'yellowgreen',
1.5 * 2 ** 13: 'green',
}

for profile in [0, 1]:
    for narray in [2 ** 12, 1.5 * 2 ** 12, 2 ** 13, 1.5 * 2 ** 13]:
        sigmas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
        perf = []
        for sigma in sigmas:
            query = '(narray == %f) & (sigma == %f) & (profile == %d)' % (narray, sigma, profile)
            samples = df.query(query)
            mac_per_cycle = np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12
            perf.append(mac_per_cycle)

        plt.plot(sigmas, perf, marker='.', color=color[narray])
        plt.ylim(bottom=0, top=21)
        plt.yticks([0, 5, 10, 15, 20], ['', '', '', ''])
        plt.xticks([0.05, 0.10, 0.15], ['', '', ''])
        plt.grid(True, linestyle='dotted')
        plt.gcf().set_size_inches(3.5, 1.5)
        plt.tight_layout(0.)
        plt.savefig('%d.png' % (profile))

    plt.cla()

######################################


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
