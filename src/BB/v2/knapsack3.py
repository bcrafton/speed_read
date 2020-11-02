
import numpy as np
import copy

############################

# https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html

capacity = 750
weight = np.array([70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120], dtype=np.float)
value = np.array([135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240], dtype=np.float)
N = len(value)
truth = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

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
        lower_bound = max(lower_bound, self.lower_bound())
        branches = []

        # try case 0
        case0 = BB(value, weight, capacity, copy.copy(self.items).append(0))
        if case0.upper_bound() >= lower_bound:
            branches.append(case0)

        # try case 1
        case1 = BB(value, weight, capacity, copy.copy(self.items).append(1))
        if case1.upper_bound() >= lower_bound:
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

    for item in range(N):
        branches = branch_and_bound_help(branches, lower_bound)
        for branch in branches:
            lower_bound = max(lower_bound, branch.lower_bound())

    return branches
        
############################

branches = branch_and_bound()

############################



















