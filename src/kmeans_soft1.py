
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
    max_sar = int(np.ceil(max_sar))

    ###################################

    choices = []
    for s in range(max_sar):
        choice = [a * (adc+1) ** s for a in range(adc + 1)]
        choices.append(choice)

    refs = []
    for comb in itertools.combinations(choices, sar):
        comb_refs = []
        for comb_ref in comb:
            if comb_refs: comb_refs = list(itertools.product(comb_ref, comb_refs))
            else:         comb_refs = comb_ref

        ref = []
        for comb_ref in comb_refs:
            ref.append( np.sum(flatten(comb_ref)) )
        refs.append(ref)

    print (refs)

###########################################################

counts = np.random.randint(low=0, high=100, size=65)
counts[27:] = 0
thresholds_kmeans_soft(counts, 1, 2)

###########################################################

