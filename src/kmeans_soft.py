
import numpy as np
import itertools
import math

###########################################################

def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)

###########################################################

def thresholds_kmeans_soft(counts, adc, sar):
    max_rpr = len(counts) - 1
    max_sar = math.log(max_rpr, adc + 1)
    # assert (max_sar % 1 == 0)
    # max_sar = int(max_sar)
    max_sar = int(np.ceil(max_sar))
    ###################################
    refs = []
    for s in range(max_sar):
        ref = [a * (adc+1) ** s for a in range(adc + 1)]
        if refs: refs = list(itertools.product(ref, refs))
        else:    refs = ref
    for i, l in enumerate(refs):
        print (i, flatten(l))

###########################################################

counts = np.random.randint(low=0, high=100, size=65)
thresholds_kmeans_soft(counts, 2, 1)

###########################################################

