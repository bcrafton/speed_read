
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

mean = {}
error = {}

for id in [0, 1, 2, 3, 4, 5]:

    error[id] = []
    mean[id] = []
    
    for sigma in sigmas:
        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (id == %d)' % ('static', 1, 1, sigma, id)
        samples = df.query(query)

        e = np.average(samples['error'])
        error[id].append(e)

        m = np.average(samples['mean'])
        mean[id].append(m)

####################

plt.cla()
for key in error:
  plt.plot(sigmas, error[key], marker='.', label=str(key))
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.legend()
plt.savefig('cc_error.png', dpi=500)

######################################

plt.cla()
for key in mean:
  plt.plot(sigmas, mean[key], marker='.', label=str(key))
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.legend()
plt.savefig('cc_mean.png', dpi=500)

####################



















