
import numpy as np
import copy

##########################################################

def array_allocation(cycles, costs, resources):
    # assert: at least 1 copy of each
    assert (np.sum(costs) <= resources)
    # start at 1 copy of each
    alloc = np.ones_like(costs)
    # find the slowest array
    argmax = np.argmax(cycles / alloc)
    # while we can allocate 1 copy to slowest array
    while (np.sum(alloc * costs) + costs[argmax]) <= resources:
        # allocate
        alloc[argmax] += 1
        # find next slowest
        argmax = np.argmax(cycles / alloc)
    return alloc

##########################################################

# do just layers rn
def perf(counts, costs, resources):
    assert (len(counts) == len(costs))
    layers = len(counts)
    cycles = []
    for (cost, count) in zip(costs, counts):
        # N, NWL, XB, WB, SIZE = np.shape(count)
        cycle = np.sum(count > 0)
        cycles.append(cycle)

    alloc = array_allocation(cycles=cycles, costs=costs, resources=resources)
    print (alloc)
    return 0

##########################################################

