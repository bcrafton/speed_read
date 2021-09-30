
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perf import compute_cycles

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

'''
print (df['thresh'])
print (df['lrs'])
print (df['hrs'])
print (df['cards'])
'''

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

hrss = [0.01]
lrss = [0.02, 0.04, 0.05, 0.06, 0.08]
perf = {}
power = {}
error = {}

for method in ['kmeans']:
    perf[method]  = []
    power[method] = []
    error[method] = []

    for hrs in hrss:
        for lrs in lrss:
            e = 0.
            top_per_sec = 0.
            top_per_pJ = 0.

            query = '(lrs == %f) & (hrs == %f) & (method == "%s")' % (lrs, hrs, method)
            samples = df.query(query)
            assert (len(samples) > 0)

            e += np.average(samples['error'])

            '''
            adcs = samples['adc']
            adc = []
            for a in adcs:
                adc.append(np.sum(a, axis=(0, 1, 2)))
            adc = np.stack(adc, axis=0)
            comps = np.arange(np.shape(adc)[-1])
            energy = np.sum(comps * adc * comp_pJ * (256 / 8), axis=1)
            top_per_pJ += 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)
            '''

            top_per_pJ = 0.

            ################################################

            # print (cards, thresh, lrs)
            # print (samples)
            counts = []
            costs = []
            layers = np.array(samples['id'])
            for layer in layers:
                query = '(id == %d)' % (layer)
                data = samples.query(query)
                #
                count = data['bb_cycles'].values[0]
                counts.append(count)
                #
                nwl = data['nwl'].values[0]
                nbl = data['nbl'].values[0]
                cost = nwl * nbl
                costs.append(cost)

            cycles = compute_cycles(counts=counts, costs=costs, resources=2**12)

            ################################################

            mac = np.sum( samples['nmac'].values )
            top_per_sec = mac / cycles
  
            ################################################

            perf[method].append(top_per_sec)
            error[method].append(e)
            power[method].append(top_per_pJ)

######################################

plt.cla()
for key in error:
  plt.plot(lrss, error[key], marker='.', label=key)
###############################
plt.ylim(bottom=0)
#plt.yticks([1, 2, 3], ['', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
###############################
#plt.ylim(bottom=0, top=17.5)
#plt.yticks([5, 10, 15], ['', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
###############################
plt.grid(True, linestyle='dotted')
#plt.gcf().set_size_inches(3.3, 1.)
#plt.tight_layout(0.)
plt.legend()
plt.savefig('cc_error.png', dpi=500)

####################

# static = np.array(perf[(1, 1, 'static')])
# skip = np.array(perf[(1, 0, 'dynamic')])
# print (static / skip)

plt.cla()
for key in perf:
  plt.plot(lrss, perf[key], marker='.', markersize=3, linewidth=1, label=key)
plt.ylim(bottom=0)
#plt.yticks([5, 10, 15, 20], ['', '', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
#plt.gcf().set_size_inches(3.3, 1.)
#plt.tight_layout(0.)
plt.legend()
plt.savefig('cc_perf.png', dpi=500)

####################

plt.cla()
for key in power:
  plt.plot(lrss, power[key], marker='.', markersize=3, linewidth=1, label=key)
plt.ylim(bottom=0)
#plt.yticks([2, 4, 6, 8], ['', '', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
#plt.gcf().set_size_inches(3.3, 1.)
#plt.tight_layout(0.)
plt.legend()
plt.savefig('cc_power.png', dpi=500)

####################





















