
import numpy as np
import copy

############################

table = np.load('table.npy', allow_pickle=True).item()
value = table['value']
weight = table['weight']
capacity = 1.
row, col = np.shape(value)

'''
64 * 64 = 4096
so we need to pick (1 / 64) for every 64.

our greedy solution can just be what we already do.

so we really want 2d array.
[64, 64]
'''

############################

class BB:

    def __init__(self, value, weight, capacity, items):
        self.value    = value
        self.weight   = weight
        self.capacity = capacity
        self.row, self.col = np.shape(self.value)

        self.items = items

    def lower_bound(self):
        total_weight = 0
        total_value = 0

        for i, item in enumerate(self.items):
            total_value += self.value[i][item]
            total_weight += self.weight[i][item]

        for i in range(len(self.items), self.row):
            for j in range(self.col):
                if total_weight + self.weight[i][j] > 0.015625:
                    total_weight += self.weight[i][j]
                    total_value += self.value[i][j]

        return total_value

    def upper_bound(self):
        total_weight = 0
        total_value = 0

        item = len(self.items)

        for i in range(0, item):
            if self.items[i]:
                total_value += self.value[i]
                total_weight += self.weight[i]

        for i in range(item, self.row):
            if total_weight + self.weight[i] < self.capacity:
                total_weight += self.weight[i]
                total_value += self.value[i]
            else:
                remainder = self.capacity - total_weight
                total_value += self.value[i] * (remainder / self.weight[i])
                total_weight += remainder

        return total_value

    def total_weight(self):
        total_weight = 0
        for i in range(len(self.items)):
            if self.items[i]:
                total_weight += self.weight[i]
        return total_weight

    def total_value(self):
        total_value = 0
        for i in range(len(self.items)):
            if self.items[i]:
                total_value += self.value[i]
        return total_value

    def solution(self):
        return np.array(self.items)

    def valid(self, lower_bound):
        if self.total_weight() > self.capacity:
            return False
        if lower_bound > self.upper_bound():
            return False
        return True

    def branch(self, lower_bound):
        item = len(self.items)
        lower_bound = max(lower_bound, self.lower_bound())
        branches = []

        # try case 0
        items0 = copy.copy(self.items)
        items0.append(0)
        case0 = BB(value, weight, capacity, items0)
        if case0.valid(lower_bound):
            branches.append(case0)

        # try case 1
        items1 = copy.copy(self.items)
        items1.append(1)
        case1 = BB(value, weight, capacity, items1)
        if case1.valid(lower_bound):
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

    for item in range(row):
        branches = branch_and_bound_help(branches, lower_bound)
        for branch in branches:
            lower_bound = max(lower_bound, branch.lower_bound())

    champ_value = 0
    champ_branch = None
    for branch in branches:
        branch_value = branch.lower_bound()
        if branch_value > champ_value:
            champ_branch = branch
            champ_value = branch_value

    return champ_branch

############################

solution = branch_and_bound()

############################















