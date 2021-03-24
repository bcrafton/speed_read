
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

SAR = False
if SAR:
    results = np.load('../results64.npy', allow_pickle=True)
else:
    results = np.load('../results8.npy', allow_pickle=True)

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

# hrs = np.unique(df['hrs'])
# lrs = np.unique(df['lrs'])
# print (hrs, lrs)

for cards, thresh in [(0, 0.25), (1, 0.25)]:
    for lrs in [0.035, 0.05, 0.10]:
        for hrs in [0.03, 0.015]:

            query = '(cards == %d) & (thresh == %f) & (lrs == %f) & (hrs == %f)' % (cards, thresh, lrs, hrs)
            samples = df.query(query)
            
            print ()
            print (cards, thresh, lrs, hrs)
            # print (len(samples))

            '''
            samples = pd.DataFrame.copy(df)
            samples = samples[close(samples['cards'],  cards)]
            samples = samples[close(samples['thresh'], thresh)]
            samples = samples[close(samples['lrs'],    lrs)]
            samples = samples[close(samples['hrs'],    hrs)]
            # print (len(samples))
            '''

            total_mac = np.sum(samples['nmac'])

            perf = 0.
            total_cycle = 0.
            for layer in range(20):

                alloc      = np.array(samples['block_alloc'])[layer].reshape(-1, 1)
                rpr        = np.array(samples['rpr'])[layer]
                block_size = np.array(samples['block_size'])[layer]
                adc        = np.array(samples['adc'])[layer] # 8x8xNWLxADC
                mac        = np.array(samples['nmac'])[layer]

                # sar = 1 + np.ceil(np.log2(rpr).reshape(8, 8, 1, 1))
                sar = 1 + np.ceil(np.log2(np.arange(1, np.shape(adc)[-1])))
                sar = np.array([0] + sar.tolist())
                if SAR: cycle = np.sum(adc * sar)
                else:   cycle = np.sum(adc)

                mac_per_cycle = mac / cycle
                perf += mac_per_cycle * (mac / total_mac)
                total_cycle += np.sum(adc, axis=(0,1,2))

                # print (np.average(rpr[0:4, :]))
                # print (rpr)


            print (perf)
            # print (np.around(total_cycle / np.sum(total_cycle), 3))




















