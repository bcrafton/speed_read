
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

for cards, thresh in [(1, 0.25), (0, 0.10)]:
    for lrs in [0.035, 0.05, 0.10]:
        for hrs in [0.05, 0.03, 0.02]:

            query = '(cards == %d) & (thresh == %f) & (lrs == %f) & (hrs == %f)' % (cards, thresh, lrs, hrs)
            samples = df.query(query)

            top_per_sec = 2. * np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12
            
            e = np.average(samples['error'])

            adc = np.stack(samples['adc'], axis=0)    
            energy = np.sum(np.arange(1, 64+1) * adc * comp_pJ, axis=1)
            top_per_pJ = 2. * np.sum(samples['nmac']) / 1e12 / np.sum(energy)

            print (cards, lrs, hrs, e, np.max(samples['cycle']), np.sum(samples['nmac']), top_per_sec, top_per_pJ)

######################################






















