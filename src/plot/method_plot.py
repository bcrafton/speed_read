
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

####################

comp_pJ = 22. * 1e-12 / 32. / 16.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

num_example = 1
sigmas = [0.02, 0.04, 0.06, 0.08, 0.10]
methods = ['kmeans', 'normal', 'soft']
perf = {}
error = {}

for method in methods:
    perf[method]  = []
    error[method] = []

    for sigma in sigmas:
        e = 0.
        top_per_sec = 0.

        for example in range(num_example):
            query = '(method == "%s") & (lrs == %f)' % (method, sigma)
            samples = df.query(query)

            adc = samples['VMM_WL']
            sar = samples['sar']
            total_cycle = 0
            for l in adc.keys():
                total_cycle += np.sum(sar[l][:, :] * np.sum(adc[l][:, :, 1:], axis=-1))

            top_per_sec += 2. * np.sum(samples['nmac']) / total_cycle * 100e6 / 1e12
            # print (samples['error'])
            e += np.average(samples['error'])

        perf[method].append(top_per_sec / num_example)
        error[method].append(e / num_example)


######################################

color = {
'kmeans': 'green',
'soft':   'blue',
'normal': 'black',
}

######################################

plt.cla()
for key in error:
  plt.plot(sigmas, error[key], color=color[key], marker='.', label=key)
plt.legend()
plt.show()

####################

plt.cla()
for key in perf:
  plt.plot(sigmas, perf[key], color=color[key], marker='.', label=key)
plt.legend()
plt.show()

####################





















