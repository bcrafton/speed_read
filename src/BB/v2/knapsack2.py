
import numpy as np
import copy

############################

capacity = 165
weight = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82], dtype=np.float)
value = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72], dtype=np.float)
N = len(value)
truth = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

############################

class BB:

    def __init__(self, value, weight, capacity, items):
        self.value    = np.array(value)
        self.weight   = np.array(weight)
        self.capacity = capacity

        self.ratio = self.value / self.weight
        order = np.argsort(self.ratio)
        self.value = self.value[order]
        self.weight = self.weight[order]

        self.N = len(self.value)
        self.items = []

    def lower_bound(self):
        total_weight = 0
        total_value = 0

        item = len(self.items)

        for i in range(0, item):
            if self.items[i]:
                total_value += self.value[i]
                total_weight += self.weight[i]

        for i in range(item, self.N):
            if total_weight + self.weight[i] < self.capacity:
                total_weight += self.weight[i]
                total_value += self.value[i]

        return total_value

    def upper_bound(self):
        total_weight = 0
        total_value = 0

        item = len(self.items)

        for i in range(0, item):
            if self.items[i]:
                total_value += self.value[i]
                total_weight += self.weight[i]

        for i in range(item, self.N):
            if total_weight + self.weight[i] < self.capacity:
                total_weight += self.weight[i]
                total_value += self.value[i]
            else:
                remainder = self.capacity - total_weight
                total_value += self.value[i] * (remainder / self.weight[i])
                total_weight += remainder

        return total_value

    def branch(self, lower_bound):
        item = len(self.items)     
        lower_bound = min(lower_bound, self.lower_bound())
        branches = []

        # try case 0
        case0 = BB(value, weight, capacity, copy.copy(self.items).append(0))
        if case0.upper_bound() <= lower_bound:
            branches.append(case0)

        # try case 1
        case1 = BB(value, weight, capacity, copy.copy(self.items).append(1))
        if case1.upper_bound() <= lower_bound:
            branches.append(case1)

        return branches

############################

def branch_and_bound_help(branches, lower_bound):
    new_branches = []
    for branch in branches:
        new_branches.extend(branch.branch(lower_bound))
    return new_branches

def branch_and_bound():
    root = BB(value, weight, capacity, [])
    branches = [root]
    lower_bound = root.lower_bound()
    for _ in range(len(weight)):
        branches = branch_and_bound_help(branches, lower_bound)
        for branch in branches:
            lower_bound = min(lower_bound, branch.lower_bound())
    return branches
        
############################

branches = branch_and_bound()
print (len(branches))

############################



















