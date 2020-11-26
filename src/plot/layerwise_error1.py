
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

results = np.load('../../results/cifar_results_perf.npy', allow_pickle=True)
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

#####################

layers = np.array(range(8))

#####################

sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

mean = {}
error = {}

for sigma in sigmas:

    error[sigma] = []
    mean[sigma] = []
    
    for id in range(8):

        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (id == %d) & (thresh == 0.1)' % ('static', 1, 1, sigma, id)
        samples = df.query(query)

        e = np.clip( np.average(samples['error']) - 0.03, 0.0, 0.1)
        error[sigma].append(e)

        m = np.average(samples['mean'])
        mean[sigma].append(m)

plt.bar(x=layers - 0.3, height=error[0.2], width=0.2, color='#808080')

#####################

sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

mean = {}
error = {}

for sigma in sigmas:

    error[sigma] = []
    mean[sigma] = []
    
    for id in range(8):

        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (id == %d) & (thresh == 0.5)' % ('static', 1, 1, sigma, id)
        samples = df.query(query)

        e = np.clip( np.average(samples['error']) - 0.03, 0.0, 0.5)
        error[sigma].append(e)

        m = np.average(samples['mean'])
        mean[sigma].append(m)

print (error[0.2])
plt.bar(x=layers - 0.1, height=error[0.2], width=0.2, color='#404040')

#####################

sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

mean = {}
error = {}

for sigma in sigmas:

    error[sigma] = []
    mean[sigma] = []
    
    for id in range(8):

        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (id == %d) & (thresh == 1)' % ('static', 1, 1, sigma, id)
        samples = df.query(query)

        e = np.clip( np.average(samples['error']) - 0.03, 0.0, 1)
        error[sigma].append(e)

        m = np.average(samples['mean'])
        mean[sigma].append(m)

plt.bar(x=layers + 0.1, height=error[0.2], width=0.2, color='#000000')

#####################

sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

mean = {}
error = {}

for sigma in sigmas:

    error[sigma] = []
    mean[sigma] = []
    
    for id in range(8):

        query = '(rpr_alloc == "%s") & (skip == %d) & (cards == %d) & (sigma == %f) & (id == %d)' % ('dynamic', 1, 0, sigma, id)
        samples = df.query(query)

        e = np.average(samples['error'])
        error[sigma].append(e)

        m = np.average(samples['mean'])
        mean[sigma].append(m)

plt.bar(x=layers + 0.3, height=error[0.2], width=0.2, color='blue')

#####################

# plt.show()

plt.ylim(bottom=-0.1, top=9.25)
plt.yticks([1, 3, 5, 7, 9], 5 * [''])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], 8 * [''])
plt.grid(True, linestyle='dotted', axis='y')
plt.gcf().set_size_inches(3.3, 1.0)
plt.tight_layout(0.)
plt.savefig('cc_error1.png', dpi=500)















