
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

SAR = True
if SAR: 
    results = np.load('../results50.npy', allow_pickle=True)
    comp_pJ = 20e-15
else:
    results = np.load('../results.npy', allow_pickle=True)
    comp_pJ = 50e-15

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

print (len(df) / 20)
print (df)

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

num_example = 1
hrss = [0.03]
lrss = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12]
perf = {}
power = {}
error = {}
stds = {}

for skip, cards, rpr_alloc, thresh in [(1, 1, 'static', 0.5)]:
    perf[(skip, cards, rpr_alloc, thresh)]  = []
    power[(skip, cards, rpr_alloc, thresh)] = []
    error[(skip, cards, rpr_alloc, thresh)] = []
    stds[(skip, cards, rpr_alloc, thresh)] = []

    for hrs in hrss:
        for lrs in lrss:
            e = 0.
            std = 0.
            top_per_sec = 0.
            top_per_pJ = 0.

            for example in range(num_example):
                # query = '(skip == %d) & (cards == %d) & (lrs == %f) & (hrs == %f) & (thresh == %f)' % (skip, cards, lrs, hrs, thresh)
                query = '(cards == %d) & (lrs == %f) & (hrs == %f) & (thresh == %f)' % (cards, lrs, hrs, thresh)
                samples = df.query(query)
                print (query)

                e += np.average(samples['error'])
                std += np.average(samples['std'])

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
                    steps = samples['step']
                    alloc = samples['block_alloc']
                    block_size = samples['block_size']
                    tops = []
                    for l in adc.keys():
                        #################################################
                        XB, WB, NWL, COMPS = np.shape(adc[l])
                        sar = np.arange(1, COMPS)
                        sar = np.minimum(sar, COMPS-1)
                        sar = 1 + np.floor(np.log2(sar))
                        sar = np.array([0] + sar.tolist())
                        #################################################
                        sar = np.reshape(sar, (1, 1, 1, COMPS))
                        step = np.reshape(steps[l], (XB, WB, 1, 1))
                        sar = np.maximum(1, sar / step)
                        #################################################
                        cycle = np.sum(adc[l] * sar)
                        #################################################
                        cycle = cycle / np.sum(alloc[l]) / block_size[l]
                        #################################################
                        max_cycle = max(max_cycle, cycle)
                        #################################################
                        # print (rpr[l].flatten())
                        # print (steps[l].flatten())
                        #################################################
                        # hist = np.sum(adc[l], axis=(0,1,2))
                        # pmf = hist / np.sum(hist)
                        # print (np.around(pmf * 100))
                        #################################################
                        # top = np.array(samples['nmac'][l]) / cycle
                        # tops.append(top)
                    ################################################
                    top_per_sec += 2. * np.sum(samples['nmac']) / max_cycle * 100e6 / 1e12
                    ################################################
                else:
                    max_cycle = 0
                    adc = samples['adc']
                    rpr = samples['rpr']
                    steps = samples['step']
                    alloc = samples['block_alloc']
                    block_size = samples['block_size']
                    tops = []
                    for l in adc.keys():
                        #################################################
                        cycle = np.sum(adc[l][..., 1:])
                        #################################################
                        cycle = cycle / np.sum(alloc[l]) / block_size[l]
                        #################################################
                        max_cycle = max(max_cycle, cycle)
                    ################################################
                    top_per_sec += 2. * np.sum(samples['nmac']) / max_cycle * 100e6 / 1e12
                    ################################################
                    
            perf[(skip, cards, rpr_alloc, thresh)].append(top_per_sec / num_example)
            error[(skip, cards, rpr_alloc, thresh)].append(e / num_example)
            power[(skip, cards, rpr_alloc, thresh)].append(top_per_pJ / num_example)
            stds[(skip, cards, rpr_alloc, thresh)].append(std / num_example)

######################################

# perf1 = np.array(perf[(1, 1, 'static', 0.25)]) / np.array(perf[(1, 0, 'static', 0.25)])
# print (perf1)
# perf1 = np.array(perf[(1, 1, 'static', 0.10)]) / np.array(perf[(1, 0, 'static', 0.25)])
# print (perf1)

color = {
(0, 0, 'dynamic', 0.10): 'green',
(1, 0, 'dynamic', 0.10): 'blue',
(1, 1, 'static', 0.10):  '#808080',
(1, 1, 'static', 0.25):  '#404040',
(1, 1, 'static', 0.50):  '#000000',
}

######################################

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

plt.cla()
for key in stds:
  plt.plot(lrss, stds[key], marker='.')
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
plt.savefig('cc_std.png', dpi=500)

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





















