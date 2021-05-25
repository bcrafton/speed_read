
import numpy as np
import copy
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

####################
'''
query = '(id == 0)'
samples = df.query(query)
bb = samples['bb'][0]

print (np.shape(bb))
'''
######################################

narray = df['narray'][0]
cycles = []
cost = []

for layer in range(8):
    cycle = np.sum(df['bb'][layer], axis=(0, 2, 3, 4))
    cycles.extend(cycle.tolist())

    WL, _, BL, _ = df['shape'][layer]
    cost.extend([BL] * WL)

cycles = np.array(cycles)
cost = np.array(cost)

######################################

def array_allocation(narray, cycle_list, cost_list):
    if (np.sum(cost_list) > narray): return np.inf
    alloc = np.ones_like(cost_list)
    cycles = cycle_list / alloc
    argmax = np.argmax(cycles)
    while (np.sum(alloc * cost_list) + cost_list[argmax]) <= narray:
        alloc[argmax] += 1
        cycles = cycle_list / alloc
        argmax = np.argmax(cycles)
    return alloc

######################################

allocation = array_allocation(narray=narray, cycle_list=cycles, cost_list=cost)
# print (allocation)
# print (cycles)
# print (cost)

######################################

print (cycles)
print (allocation)
print (np.sum(allocation * cost))

def simulate(cycles, allocation):
    t = 0
    while not np.all(cycles == 0):
        cycles = np.maximum(cycles - allocation, 0)
        t += 1
    return t

t = simulate(cycles, allocation)
print (t)

######################################







