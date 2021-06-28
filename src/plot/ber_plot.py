
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
# print (df['ecc'])
# print (df['example'])

######################################

colors = ['#228B22', '#96F078', '#ffae42', '#ff849c', '#ff0000']
layers = np.arange(8, dtype=int)

sigma = 0.07
N = 3

for sigma in [0.01, 0.07, 0.085, 0.10]:
    for ecc in [0, 1]:    
        query = '(sigma == %f) & (ecc == %d)' % (sigma, ecc)
        samples = df.query(query)
        ber = np.array(samples['BER'])
        print (sigma, ecc, np.min(ber), np.mean(ber), np.median(ber), np.max(ber))

plt.yscale('log')
plt.ylim(bottom=0.25, top=10**9)
plt.yticks(10**np.array([1,3,5,7]), 4 * [''])

plt.xticks(layers, len(layers) * [''])

plt.gcf().set_size_inches(3.5, 0.8)
plt.grid(True, axis='y', linestyle='dotted', color='black')
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('error_rate_%f.png' % (sigma), dpi=500)

######################################















