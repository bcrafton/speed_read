
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
results = ld_to_dl(results)

####################

df = pd.DataFrame.from_dict(results)

####################

comp_pJ = 20e-15

hrss = [0.03]
lrss = [0.02, 0.04, 0.06, 0.08]
perf = {}
power = {}
error = {}

for cards, thresh, method in [(0, 0.10, 'normal'), (0, 0.10, 'kmeans'), (1, 0.10, 'normal'), (1, 0.10, 'kmeans')]:
    perf[(cards, thresh, method)] = []
    error[(cards, thresh, method)] = []

    for hrs in hrss:
        for lrs in lrss:
            ##################################################################
            query = '(cards == %d) & (lrs == %f) & (hrs == %f) & (thresh == %f) & (method == "%s")' % (cards, lrs, hrs, thresh, method)
            samples = df.query(query)
            ##################################################################
            total_cycle = 0
            count = samples['count']
            rpr = samples['rpr']
            steps = samples['step']
            tops = []
            for l in count.keys():
                #################################################
                N, NWL, XB, WB, SIZE = np.shape(count[l])
                adc = count[l].transpose(2, 3, 0, 1, 4).reshape(XB, WB, N * NWL * SIZE)
                cycle = 0
                # print (steps[l])
                # print (rpr[l])
                for i in range(XB):
                    for j in range(WB):
                        #################################################
                        values, counts = np.unique(adc[i][j], return_counts=True)
                        sar = np.where(values > 0, 1 + np.ceil(np.log2(values)),        0)
                        sar = np.where(sar    > 0, np.maximum(1, sar - steps[l][i][j]), 0)
                        total_cycle += np.sum(sar * counts)
                        #################################################
            top_per_sec = total_cycle
            ##################################################################
            e = np.max(samples['error'])
            ##################################################################
            perf[(cards, thresh, method)].append(top_per_sec)
            error[(cards, thresh, method)].append(e)

######################################

print (perf)
print (error)

######################################

color = {
(0, 0.1): 'blue',
(1, 0.10): '#808080',
(1, 0.25): '#404040',
(1, 0.50): '#000000',
}

######################################
'''
plt.cla()
for key in perf:
  plt.plot(lrss, perf[key], color=color[key], marker='.', markersize=3, linewidth=1)
# plt.ylim(bottom=0, top=32.5)
# plt.yticks([0, 10, 20, 30], ['', '', '', ''])
#
# plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
# plt.gcf().set_size_inches(1.65, 1.)
# plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)
'''
####################
'''
plt.cla()
for key in perf:
  plt.plot(lrss, error[key], color=color[key], marker='.', markersize=3, linewidth=1)
# plt.ylim(bottom=0, top=32.5)
# plt.yticks([0, 10, 20, 30], ['', '', '', ''])
#
# plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
# plt.gcf().set_size_inches(1.65, 1.)
# plt.tight_layout(0.)
plt.savefig('cc_error.png', dpi=500)
'''
####################


















