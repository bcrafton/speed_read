
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
lrss = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
areas = [64, 128, 192, 256, 384, 512, 640, 768, 896, 1024]

perf = {}
power = {}
error = {}

for area in areas:
    for hrs in hrss:
        for lrs in lrss:
            perf[(lrs, hrs, area)] = []
            error[(lrs, hrs, area)] = []
            ##################################################################
            query = '(lrs == %f) & (hrs == %f) & (area == %d)' % (lrs, hrs, area)
            samples = df.query(query)
            ##################################################################
            total_wl = 0
            total_cycle = 0
            hist = np.zeros(shape=33)

            count = samples['count']
            rpr = samples['rpr']
            steps = samples['step']
            sar = samples['sar']
            comps = samples['comps']
            tops = []

            for l in count.keys():
                if samples['id'][l] != 1: continue
                # print (area, N, lrs)
                # print (sar[l])
                # print (rpr[l])
                # print (comps[l])
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
                        #################################################
                        for v, s, c in zip(values, scale, counts):
                            total_wl += v * c
                            hist[int(s)] += c
                        #################################################
            top_per_sec = total_cycle / N
            ##################################################################
            e = np.max(samples['error'])
            ##################################################################
            perf[(cards, thresh, method, N, area)].append(top_per_sec)
            error[(cards, thresh, method, N, area)].append(e)
            ##################################################################

######################################

plt.cla()
for key in perf:
  plt.plot(lrss, perf[key], color=color[key[3]], marker='.', markersize=3, linewidth=1, label=[str(x) for x in key])
  print (key, perf[key])
print ()

# plt.ylim(bottom=0, top=32.5)
# plt.yticks([0, 10, 20, 30], ['', '', '', ''])
#
# plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
# plt.gcf().set_size_inches(1.65, 1.)
# plt.legend()
# plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)

####################

plt.cla()
for key in perf:
  plt.plot(lrss, error[key], color=color[key[3]], marker='.', markersize=3, linewidth=1, label=[str(x) for x in key])
  print (key, error[key])
print ()

# plt.ylim(bottom=0, top=32.5)
# plt.yticks([0, 10, 20, 30], ['', '', '', ''])
#
# plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
# plt.gcf().set_size_inches(1.65, 1.)
# plt.legend()
# plt.tight_layout(0.)
plt.savefig('cc_error.png', dpi=500)

####################


















