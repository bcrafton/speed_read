
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

results = np.load('../results_flash.npy', allow_pickle=True)

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

####################

comp_pJ = 45e-15

hrss = [0.015]
lrss = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
perf = {}
power = {}
error = {}

for cards, thresh in [(0, 0.25), (1, 0.1), (1, 0.5), (1, 0.25)]:
    perf[(cards, thresh)]  = []
    power[(cards, thresh)] = []
    error[(cards, thresh)] = []

    for hrs in hrss:
        for lrs in lrss:
            ##################################################################
            query = '(cards == %d) & (lrs == %f) & (hrs == %f) & (thresh == %f)' % (cards, lrs, hrs, thresh)
            samples = df.query(query)
            ##################################################################
            max_cycle = 0
            adc = samples['adc']
            rpr = samples['rpr']
            steps = samples['step']
            alloc = samples['block_alloc']
            block_size = samples['block_size']
            tops = []
            for l in adc.keys():
                cycle = np.sum(adc[l][..., 1:])
                cycle = cycle / np.sum(alloc[l]) / block_size[l]
                max_cycle = max(max_cycle, cycle)
            top_per_sec = 2. * np.sum(samples['nmac']) / max_cycle * 100e6 / 1e12
            ##################################################################
            e = np.average(samples['error'])
            ##################################################################
            adcs = samples['adc']
            adc = []
            for a in adcs:
                adc.append(np.sum(a, axis=(0, 1, 2)))
            adc = np.stack(adc, axis=0)
            comps = np.arange(np.shape(adc)[-1])
            energy = np.sum(comps * adc * comp_pJ * (256 / 8), axis=1)

            energy += samples['ron'] * 2e-16
            energy += samples['roff'] * 2e-16
            top_per_pJ = 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)
            ##################################################################

            perf[(cards, thresh)].append(top_per_sec)
            error[(cards, thresh)].append(e)
            power[(cards, thresh)].append(top_per_pJ)

######################################

plot = {'perf': perf, 'power': power}
np.save('flash_plot.npy', plot)

######################################

color = {
(0, 0.25): 'green',
(0, 0.25): 'blue',
(1, 0.10): '#808080',
(1, 0.25): '#404040',
(1, 0.50): '#000000',
}

######################################

plt.cla()
for key in perf:
  plt.plot(lrss, perf[key], color=color[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0, top=32.5)
plt.yticks([0, 10, 20, 30], ['', '', '', ''])
#
plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(1.65, 1.)
plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)

####################

plt.cla()
for key in power:
  plt.plot(lrss, power[key], color=color[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0, top=9.)
plt.yticks([2, 4, 6, 8], ['', '', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(1.65, 1.)
plt.tight_layout(0.)
plt.savefig('cc_power.png', dpi=500)

####################





















