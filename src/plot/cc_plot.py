
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

###############################################################

# block  = NWL * [DUPLICATE]
# vector = [ROW, NWL]
def compute_cycles(alloc, vector):
    ROW, NWL = np.shape(vector)
    assert (np.shape(alloc) == (NWL,))
    #################################################
    block_cycles = [np.zeros(COPY) for COPY in alloc]
    #################################################
    for row in range(ROW):
        for block in range(NWL):
            copy = np.argmin(block_cycles[block])
            block_cycles[block][copy] += vector[row][block]
    #################################################
    cycles = np.zeros(shape=NWL)
    for block in range(NWL):
        cycles[block] = np.max(block_cycles[block])
    return np.max(cycles)

###############################################################

for cards, thresh in [(1, 0.10)]:
    for lrs in [0.035]:
        for hrs in [0.05]:

            query = '(cards == %d) & (thresh == %f) & (lrs == %f) & (hrs == %f)' % (cards, thresh, lrs, hrs)
            samples = df.query(query)

            cycles = []
            for layer in range(20):
                cycle = samples['cycle'][layer]
                print (cycle)

                alloc  = samples['block_alloc'][layer]
                block = samples['block_cycle'][layer]
                block_cycle = compute_cycles(alloc, block)
                print (block_cycle)

                #####################################################

                adc = samples['adc'][layer] # 8x8xNWLxADC
                cycle = adc / alloc.reshape(-1, 1)
                cycle = np.sum(cycle, axis=(0, 1, 3))
                cycle = np.max(cycle)
                print (cycle)

                #####################################################

                '''
                adc = samples['adc'][layer]
                vector = np.sum(adc, axis=(1,2,4))
                block_size = samples['block_size'][layer]
                cycle = compute_cycles(alloc, vector // block_size)
                print (cycle)
                '''
                
                #####################################################

                '''
                rpr = samples['rpr'][layer]
                sar = np.maximum(1, np.ceil(np.log2(rpr)))
                
                adc = samples['adc'][layer]
                vector = np.sum(adc * np.reshape(sar, (8, 8, 1, 1)), axis=(1,2,4))
                block_size = samples['block_size'][layer]
                cycle = compute_cycles(alloc, vector // block_size)
                cycles.append(cycle)
                '''

                #####################################################
            
            print (np.max(cycles))





















