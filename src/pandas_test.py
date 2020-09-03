
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

results = np.load('results.npy', allow_pickle=True)
results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

####################

print (df['stall'])

cond = df.where(df['stall'] > 1000000)
print (cond)

cond = df[ df['stall'] > 1000000 ]
print (cond)

####################

cond = df[ df['id'] == 0 ]
print (cond)

####################

cond = df[ df['id'] == 0 ][ df['alloc'] == 'block' ]
print (cond)

####################
