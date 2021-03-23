
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

# block  = NWL * [DUPLICATE]
# vector = [ROW, NWL]
def cycles(alloc, vector):
    ROW, NWL = np.shape(vector)
    assert (np.shape(alloc) == (NWL,))
    #################################################
    block_cycles = [np.zeros(COPY) for COPY in alloc]
    #################################################
    for row in range(ROW):
        for block in range(NWL):
            assert (row < 200)
            copy = np.argmin(block_cycles[block])
            # print (copy)
            # print (block)
            print (block_cycles[block])
            print (vector[row][block])
            block_cycles[block][copy] += vector[row][block]
    #################################################
    cycles = np.zeros(shape=NWL)
    for block in range(NWL):
        cycles[block] = np.max(block_cycles[block])
    return cycles

for cards, thresh in [(0, 0.10)]:
    for lrs in [0.035]:
        for hrs in [0.05]:

            query = '(cards == %d) & (thresh == %f) & (lrs == %f) & (hrs == %f)' % (cards, thresh, lrs, hrs)
            samples = df.query(query)

            top_per_sec = 2. * np.sum(samples['nmac']) / np.max(samples['cycle']) * 100e6 / 1e12
            
            e = np.average(samples['error'])

            '''
            for block in samples['block_cycle']:
                print (block)
            '''
            '''
            for adc in samples['adc']:
                cycle = np.sum(adc, axis=(0, 1, 3))
                print (cycle)
            '''
            '''
            for adc in samples['adc']:
                cycle = np.sum(adc)
                pmf = np.sum(adc, axis=(0,1,2)) / cycle
                print (pmf)
            '''
            '''
            for layer in range(20):
                cycles = np.max(samples['block_cycle'][layer] / samples['block_alloc'][layer])
                print (cycles)
                print (samples['cycle'][layer])
            '''
            for layer in range(1):
                alloc  = samples['block_alloc'][layer]
                vector = samples['block_cycle'][layer]
                c = cycles(alloc, vector)
                print (samples['cycle'][layer], c)

######################################






















