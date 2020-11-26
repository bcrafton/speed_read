
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

results = ld_to_dl(np.load('../../results/imagenet_results_perf1.npy', allow_pickle=True))
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['rpr_alloc'])
# print (df['example'])

####################

comp_pJ = 22. * 1e-12 / 32. / 16.

# power plot is problem -> [0, 0], [1, 0] produce same result.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

num_example = 1
sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
perf = {}
power = {}
error = {}

for skip, cards, rpr_alloc, thresh in [(0, 0, 'dynamic', 0.10), (1, 0, 'dynamic', 0.10), (1, 1, 'static', 0.10), (1, 1, 'static', 0.25), (1, 1, 'static', 0.50)]:
    perf[(skip, cards, rpr_alloc, thresh)]  = []
    power[(skip, cards, rpr_alloc, thresh)] = []
    error[(skip, cards, rpr_alloc, thresh)] = []

    for sigma in sigmas:
        e = 0.
        top_per_sec = 0.
        top_per_pJ = 0.

        for example in range(num_example):
            query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (example == %d) & (thresh == %f)' % (rpr_alloc, skip, cards, sigma, example, thresh)
            samples = df.query(query)

            top_per_sec += 2. * np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12

            e += np.average(samples['error'])

            adc = np.stack(samples['adc'], axis=0)    
            energy = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_pJ, axis=1) 
            energy += samples['ron'] * 2e-16
            energy += samples['roff'] * 2e-16
            top_per_pJ += 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)

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
  plt.plot(sigmas, error[key], color=color[key], marker='.')
###############################
plt.ylim(bottom=0, top=3.5)
plt.yticks([1, 2, 3], ['', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
###############################
'''
plt.ylim(bottom=0, top=17.5)
plt.yticks([5, 10, 15], ['', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
'''
###############################
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(3.3, 1.)
plt.tight_layout(0.)
plt.savefig('cc_error.png', dpi=500)

####################

# MAKE SURE PY3 IS ENABLED ...

static1 = np.array(perf[(1, 1, 'static', 0.10)])
static2 = np.array(perf[(1, 1, 'static', 0.25)])
static3 = np.array(perf[(1, 1, 'static', 0.50)])
# skip = np.array(perf[(1, 0, 'dynamic', 0.10)])

print (static3)
print (static2)
print (static1)

plt.cla()
for key in perf:
  plt.plot(sigmas, perf[key], color=color[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0, top=22.5)
plt.yticks([5, 10, 15, 20], ['', '', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(3.3, 1.)
plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)

####################

static1 = np.array(power[(1, 1, 'static', 0.10)])
static2 = np.array(power[(1, 1, 'static', 0.25)])
static3 = np.array(power[(1, 1, 'static', 0.50)])
# skip = np.array(power[(1, 0, 'dynamic', 0.10)])

print (static3)
print (static2)
print (static1)

plt.cla()
for key in power:
  plt.plot(sigmas, power[key], color=color[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0, top=9.)
plt.yticks([2, 4, 6, 8], ['', '', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(3.3, 1.)
plt.tight_layout(0.)
plt.savefig('cc_power.png', dpi=500)

####################





















