
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
# print (df['example'])

####################

comp_pJ = 22. * 1e-12 / 32. / 16.

# power plot is problem -> [0, 0], [1, 0] produce same result.

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

num_example = 1
perf = {}
power = {}
error = {}

for profile, alloc in [(0, 'block'), (1, 'block'), (0, 'layer'), (1, 'layer')]:
    perf[(profile, alloc)]  = []
    power[(profile, alloc)] = []

    query = '(profile == %d) & (alloc == "%s")' % (profile, alloc)
    samples = df.query(query)

    top_per_sec = 2. * np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12

    adc = np.stack(samples['adc'], axis=0)    
    energy = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_pJ, axis=1) 
    energy += samples['ron'] * 2e-16
    energy += samples['roff'] * 2e-16
    top_per_pJ = 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)

    print (profile, alloc)
    print (np.sum(samples['nmac']), np.max(samples['cycle']))
    print (top_per_sec * 128)
    print (top_per_pJ * 128)

    cycles        = np.array(samples['cycle'])
    nmac          = np.array(samples['nmac'])
    block_density = np.array(samples['block_density'])
    density       = np.array(samples['density'])
    # print (cycles)
    # print (nmac)

######################################




















