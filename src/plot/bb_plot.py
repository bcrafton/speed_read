
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

def ll_to_l(ll):
    l = []
    shape = []
    for x in ll:
        shape.append(len(x))
        l.extend(x)
    return l, shape

####################

def l_to_ll(l, shape):
    ll = []
    idx = [0] + np.cumsum(shape).tolist()
    for i, _ in enumerate(shape):
        start = idx[i]
        end = idx[i+1]
        ll.append(l[start:end])
    return ll

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
    cycles.append(cycle.tolist())

    WL, _, BL, _ = df['shape'][layer]
    cost.extend([BL] * WL)

cycles, shape = ll_to_l(cycles)
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
    return alloc.tolist()

######################################

allocation = array_allocation(narray=narray, cycle_list=cycles, cost_list=cost)
# print (allocation)
# print (cycles)
# print (cost)
# print (allocation * cost)
# print (np.sum(allocation * cost))

######################################

cycles = []
for layer in range(8):
    cycle = np.sum(df['bb'][layer], axis=(2, 3, 4))
    cycles.append(cycle.T)

allocation = l_to_ll(allocation, shape)

######################################    

def simulate(ops, allocation):
    assert (len(allocation) == len(ops))
    max_cycles = 0

    N = len(ops)
    for l in range(N):

        B = len(ops[l])
        for b in range(B):

            cycles = 0
            while np.sum(ops[l][b]) > 0:
                d = 0
                i = 0
                while d < allocation[l][b] and i < len(ops[l][b]):
                    if ops[l][b][i] > 0:
                        ops[l][b][i] -= 1
                        d += 1
                    i += 1
                cycles += 1
            max_cycles = max(max_cycles, cycles)
    return max_cycles

cycles = simulate(cycles, allocation)
print (cycles)

######################################

























