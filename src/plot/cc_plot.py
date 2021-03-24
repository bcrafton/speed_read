
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

results = np.load('../results16.npy', allow_pickle=True)
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

def close(a, b, tol=1e-9):
    return pd.DataFrame.abs(a - b) < tol

###############################################################

SAR = False

for cards, thresh in [(0, 0.10), (1, 0.10)]:
    for lrs in [0.035, 0.10]:
        for hrs in [0.05, 0.02]:
            print ()
            print (cards, thresh, lrs, hrs)

            query = '(cards == %d) & (thresh == %f) & (lrs == %f) & (hrs == %f)' % (cards, thresh, lrs, hrs)
            samples = df.query(query)
            # print (len(samples))
            total_mac = np.sum(samples['nmac'])

            '''
            samples = pd.DataFrame.copy(df)
            samples = samples[close(samples['cards'],  cards)]
            samples = samples[close(samples['thresh'], thresh)]
            samples = samples[close(samples['lrs'],    lrs)]
            samples = samples[close(samples['hrs'],    hrs)]
            # print (len(samples))
            '''

            perf = 0.
            for layer in range(20):

                alloc      = np.array(samples['block_alloc'])[layer].reshape(-1, 1)
                rpr        = np.array(samples['rpr'])[layer]
                block_size = np.array(samples['block_size'])[layer]
                adc        = np.array(samples['adc'])[layer] # 8x8xNWLxADC
                mac        = np.array(samples['nmac'])[layer]

                sar = 1 + np.ceil(np.log2(rpr).reshape(8, 8, 1, 1))

                if SAR: cycle = np.sum(adc * sar)
                else:   cycle = np.sum(adc * 2)
                mac_per_cycle = mac / cycle
                perf += mac_per_cycle * (mac / total_mac)
                # print (mac_per_cycle, np.average(rpr[0:4, :]))
                # print (rpr)

            print (perf)




















