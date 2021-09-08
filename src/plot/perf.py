
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

def emulate(count, alloc):
    ret = 0
    N, NWL, XB, WB, SIZE = np.shape(count)
    # for each patch (N)
    # [XB, WB, SIZE] can be summed together
    # NWL can be summed for (layer) but not (block)
    #######################################################
    cycles = np.sum(count > 0, axis=(1, 2, 3, 4))
    #######################################################
    next = 0
    mode = True
    total = len(cycles)
    mappings = -1 * np.ones(shape=alloc, dtype=int)
    #######################################################
    total_cycles = 0
    while np.sum(cycles) > 0:
        total_cycles += 1
        for arr in range(alloc):
            allocated = (mappings[arr] >= 0)
            complete = (cycles[mappings[arr]] <= 0)
            remainder = total - next
            # should not allocate an job with zero cycles
            if (not allocated or complete) and (remainder > 0):
                mappings[arr] = next
                next += 1

            stall = (remainder <= 0) and complete
            if not stall:
                cycles[mappings[arr]] -= 1

    return total_cycles

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

    allocs = array_allocation(cycles=cycles, costs=costs, resources=resources)
    # print (alloc)
    # return 0

    assert (len(counts) == len(allocs))
    cycles = []
    for (count, alloc) in zip(counts, allocs):
        cycle = emulate(count=count, alloc=alloc)
        cycles.append(cycle)
    return np.max(cycles)

##########################################################




