
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

sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
perf = {}
power = {}
error = {}

for skip, cards, rpr_alloc in [(0, 0, 'dynamic'), (1, 0, 'dynamic'), (1, 1, 'static')]:

    perf[(skip, cards, rpr_alloc)]  = []
    power[(skip, cards, rpr_alloc)] = []
    error[(skip, cards, rpr_alloc)] = []
    
    for sigma in sigmas:
        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f)' % (rpr_alloc, skip, cards, sigma)
        samples = df.query(query)
        
        mac_per_cycle = np.sum(samples['nmac']) / np.max(samples['cycle']) * 2. * 100e6 / 1e12
        perf[(skip, cards, rpr_alloc)].append(mac_per_cycle)
        
        e = np.average(samples['std'])
        error[(skip, cards, rpr_alloc)].append(e)

        adc = np.stack(samples['adc'], axis=0)    
        energy = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_pJ, axis=1) 
        energy += samples['ron'] * 2e-16
        energy += samples['roff'] * 2e-16
        mac_per_pJ = np.sum(samples['nmac']) / 1e12 / np.sum(energy)
        power[(skip, cards, rpr_alloc)].append(mac_per_pJ)
        
######################################

color = {
(0, 0, 'dynamic'): 'green',
(1, 0, 'dynamic'): 'blue',
(1, 1, 'static'):  'black',
}

plt.cla()
for key in error:
  plt.plot(sigmas, error[key], color=color[key], marker='.')
  ###############################
  # plt.ylim(bottom=0, top=3.5)
  # plt.yticks([1, 2, 3], ['', '', ''])
  # plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
  ###############################
  plt.ylim(bottom=0, top=17.5)
  plt.yticks([5, 10, 15], ['', '', ''])
  plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
  ###############################
  plt.grid(True, linestyle='dotted')
  plt.gcf().set_size_inches(3.5, 1.25)
  plt.tight_layout(0.)
  plt.savefig('cc_error.png', dpi=500)

####################

# static = np.array(perf[(1, 1, 'static')])
# skip = np.array(perf[(1, 0, 'dynamic')])
# print (static / skip)

plt.cla()
for key in perf:
  plt.plot(sigmas, perf[key], color=color[key], marker='.')
  plt.ylim(bottom=0, top=42.5)
  plt.yticks([10, 20, 30, 40], ['', '', '', ''])
  plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
  plt.grid(True, linestyle='dotted')
  plt.gcf().set_size_inches(3.5, 1.25)
  plt.tight_layout(0.)
  plt.savefig('cc_perf.png', dpi=500)

####################

plt.cla()
for key in power:
  plt.plot(sigmas, power[key], color=color[key], marker='.')
  plt.ylim(bottom=0, top=9.)
  plt.yticks([2, 4, 6, 8], ['', '', '', ''])
  plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
  plt.grid(True, linestyle='dotted')
  plt.gcf().set_size_inches(3.5, 1.25)
  plt.tight_layout(0.)
  plt.savefig('cc_power.png', dpi=500)

####################





















