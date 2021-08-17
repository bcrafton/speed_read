
import numpy as np
import matplotlib
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

'''
results1 = np.load('../results1.npy', allow_pickle=True).tolist()
results2 = np.load('../results2.npy', allow_pickle=True).tolist()
results = ld_to_dl(results1 + results2)
'''

results = np.load('../results.npy', allow_pickle=True).tolist()

####################

df = pd.DataFrame.from_dict(results)
# print (df.columns)

####################

comp_pJ = 20e-15

hrss = [0.02]
lrss = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
areas = np.array([64, 128, 192, 256, 384, 512, 640, 768, 896, 1024])

perf = np.zeros(shape=(len(areas), len(lrss)))
error = np.zeros(shape=(len(areas), len(lrss)))

sars = np.zeros(shape=(len(areas), len(lrss)))
Ns = np.zeros(shape=(len(areas), len(lrss)))
adcs = np.zeros(shape=(len(areas), len(lrss)))

for area_i, area in enumerate(areas):
    for hrs in hrss:
        for lrs_i, lrs in enumerate(lrss):
            ##################################################################
            query = '(lrs == %f) & (hrs == %f) & (area == %d)' % (lrs, hrs, area)
            samples = df.query(query)
            ##################################################################
            total_wl = 0
            total_cycle = 0
            total_mac = 0
            hist = np.zeros(shape=33)

            count = samples['count']
            rpr = samples['rpr']
            steps = samples['step']
            sar = samples['sar']
            comps = samples['comps']
            N = samples['N']

            tops = []
            for l in count.keys():
                if samples['id'][l] != 1: continue
                # print (area, N, lrs)
                # print (sar[l])
                # print (rpr[l])
                # print (comps[l])
                #################################################
                sars[area_i][lrs_i] = np.max(sar[l]) > 1
                adcs[area_i][lrs_i] = np.max(comps[l])
                Ns[area_i][lrs_i] = N[l]
                #################################################
                P, NWL, XB, WB, SIZE = np.shape(count[l])
                adc = count[l].transpose(2, 3, 0, 1, 4).reshape(XB, WB, P * NWL * SIZE)
                for i in range(XB):
                    for j in range(WB):
                        #################################################
                        values, counts = np.unique(adc[i][j], return_counts=True)
                        #################################################
                        if sar[l][i][j]:
                            scale = np.where(values > 0, 1 + np.ceil(np.log2(values)),              0)
                            scale = np.where(scale  > 0, np.maximum(1, scale - steps[l][i][j] + 1), 0)
                        else:
                            scale = np.where(values > 0, 1, 0)

                        total_cycle += np.sum(scale * counts)
                        total_mac += samples['nmac'][l]
                        #################################################
                        for v, s, c in zip(values, scale, counts):
                            total_wl += v * c
                            hist[int(s)] += c
                        #################################################
            error[area_i][lrs_i] = np.max(samples['error'])
            perf[area_i][lrs_i] = total_mac / (total_cycle / N[l])
            ##################################################################

######################################

print (adcs)
print (sars)
print (Ns)

normalize = matplotlib.colors.Normalize(vmin=0, vmax=50)
perf_plot = perf
perf_plot = perf_plot - np.min(perf_plot)
perf_plot = perf_plot / np.max(perf_plot)
perf_plot = perf_plot * 15 + 10

plt.imshow(X=perf_plot, cmap='hot', norm=normalize, aspect=0.66)

####################

perf = perf / np.min(perf)

####################

ax = plt.gca();

lrss = np.array(lrss)
# lrss = (100 * lrss).astype(int)
# labels = [str(lrs) for lrs in lrss]
labels = ['%0.2f' % (lrs) for lrs in lrss]
ticks = np.arange(len(labels), dtype=int)
plt.xticks(ticks, labels)

areas = np.array(areas)
areas = (areas / np.min(areas)).astype(int)
labels = [str(area) for area in areas]
ticks = np.arange(len(labels), dtype=int)
plt.yticks(ticks, labels)

####################

for i in range(0, 10):
  for j in range(0, 10):
      annotation = '%0.1f×' % perf[i, j]
      # annotation = '%d×%s%d' % (Ns[i, j], 'S' if sars[i, j] else 'F', adcs[i, j])
      text = plt.text(j, i, annotation, ha="center", va="center", color="black")

####################

plt.xlabel('LRS Variation')
plt.ylabel('Area Factor')
# plt.colorbar()

####################

# plt.gcf().set_size_inches(2., 1.32)
# plt.tight_layout(0.)
plt.savefig('perf.png', dpi=600)

####################














