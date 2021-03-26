
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

SAR = False
if SAR: 
    results = np.load('../results64a.npy', allow_pickle=True)
    comp_pJ = 20e-15
else:
    results = np.load('../results8a.npy', allow_pickle=True)
    comp_pJ = 50e-15

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

num_example = 1
hrss = [0.03]
lrss = [0.02, 0.04, 0.06, 0.08]
perf = {}
power = {}
error = {}

# for skip, cards, rpr_alloc, thresh in [(1, 0, 'static', 0.25), (1, 1, 'static', 0.10), (1, 1, 'static', 0.25)]:
for skip, cards, rpr_alloc, thresh in [(1, 0, 'static', 0.25)]:
    perf[(skip, cards, rpr_alloc, thresh)]  = []
    power[(skip, cards, rpr_alloc, thresh)] = []
    error[(skip, cards, rpr_alloc, thresh)] = []

    for hrs in hrss:
        for lrs in lrss:
            e = 0.
            top_per_sec = 0.
            top_per_pJ = 0.

            for example in range(num_example):
                query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (lrs == %f) & (hrs == %f) & (example == %d) & (thresh == %f)' % (rpr_alloc, skip, cards, lrs, hrs, example, thresh)
                samples = df.query(query)

                e += np.average(samples['error'])

                adcs = samples['adc']
                adc = []
                for a in adcs:
                    adc.append(np.sum(a, axis=(0, 1, 2)))
                adc = np.stack(adc, axis=0)
                comps = np.arange(np.shape(adc)[-1])
                energy = np.sum(comps * adc * comp_pJ * (256 / 8), axis=1)

                top_per_pJ += 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)

                ################################################
                if SAR:
                    max_cycle = 0
                    adc = samples['adc']
                    rpr = samples['rpr']
                    alloc = samples['block_alloc']
                    block_size = samples['block_size']
                    tops = []
                    for l in adc.keys():
                        #################################################
                        sar = np.arange(1, np.shape(adc[l])[-1])
                        sar = np.minimum(sar, np.shape(adc[l])[-1] - 2)
                        sar = 1 + np.floor(np.log2(sar))
                        sar = np.array([0] + sar.tolist())
                        #################################################
                        cycle = np.sum(adc[l] * sar)
                        #################################################
                        cycle = cycle / np.sum(alloc[l]) / block_size[l]
                        #################################################
                        max_cycle = max(max_cycle, cycle)
                        #################################################
                        # print (rpr[l])
                        #################################################
                        # hist = np.sum(adc[l], axis=(0,1,2))
                        # pmf = hist / np.sum(hist)
                        # print (np.around(pmf * 100))
                        #################################################
                        top = np.array(samples['nmac'][l]) / cycle
                        tops.append(top)
                    ################################################
                    top_per_sec += 2. * np.sum(samples['nmac']) / max_cycle * 100e6 / 1e12
                    ################################################
                else:
                    '''
                    top_per_sec += 2. * np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12
                    top = np.array(samples['nmac']) / np.array(samples['cycle'])
                    print (np.around(top))
                    '''
                    max_cycle = 0
                    adc = samples['adc']
                    rpr = samples['rpr']
                    alloc = samples['block_alloc']
                    block_size = samples['block_size']
                    tops = []
                    for l in adc.keys():
                        #################################################
                        adc[l][0] = 0
                        cycle = np.sum(adc[l])
                        #################################################
                        cycle = cycle / np.sum(alloc[l]) / block_size[l]
                        #################################################
                        max_cycle = max(max_cycle, cycle)
                        #################################################
                        top = np.array(samples['nmac'][l]) / cycle
                        tops.append(top)
                    ################################################
                    top_per_sec += 2. * np.sum(samples['nmac']) / max_cycle * 100e6 / 1e12
                    ################################################

            perf[(skip, cards, rpr_alloc, thresh)].append(top_per_sec / num_example)
            error[(skip, cards, rpr_alloc, thresh)].append(e / num_example)
            power[(skip, cards, rpr_alloc, thresh)].append(top_per_pJ / num_example)

######################################

color = {
(0, 0, 'dynamic', 0.10): 'green',
(1, 0, 'dynamic', 0.10): 'blue',
(1, 1, 'static', 0.10):  '#808080',
(1, 1, 'static', 0.25):  '#404040',
(1, 1, 'static', 0.50):  '#000000',
}

plt.cla()
for key in error:
  plt.plot(lrss, error[key], marker='.')
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
plt.savefig('cc_error.png', dpi=500)

####################

# static = np.array(perf[(1, 1, 'static')])
# skip = np.array(perf[(1, 0, 'dynamic')])
# print (static / skip)

plt.cla()
for key in perf:
  plt.plot(lrss, perf[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0)
#plt.yticks([5, 10, 15, 20], ['', '', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
#plt.gcf().set_size_inches(3.3, 1.)
#plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)

####################

plt.cla()
for key in power:
  plt.plot(lrss, power[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0)
#plt.yticks([2, 4, 6, 8], ['', '', '', ''])
#plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
#plt.gcf().set_size_inches(3.3, 1.)
#plt.tight_layout(0.)
plt.savefig('cc_power.png', dpi=500)

####################





















