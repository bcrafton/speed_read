
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
            best = 0
            for j in range(self.col):
                if total_weight + self.weight[i][j] < 0.015625:
                    best = j
            total_weight += self.weight[i][best]
            total_value += self.value[i][best]

        return total_value

    def upper_bound(self):
        total_weight = 0
        total_value = 0

        for i, item in enumerate(self.items):
            total_value += self.value[i][item]
            total_weight += self.weight[i][item]

        for i in range(len(self.items), self.row):
            best = 0
            for j in range(self.col):
                if total_weight + self.weight[i][j] < 0.015625:
                    best = min(j + 1, self.col - 1)
            total_weight += self.weight[i][best]
            total_value += self.value[i][best]

        return total_value

    def total_weight(self):
        total_weight = 0
        for i, item in enumerate(self.items):
            total_weight += self.weight[i][item]
        return total_weight

    def total_value(self):
        total_value = 0
        for i, item in enumerate(self.items):
            total_value += self.value[i][item]
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

        for item in range(self.col):
            new_items = copy.copy(self.items)
            new_items.append(item)
            new_branch = BB(self.value, self.weight, self.capacity, new_items)
            if new_branch.valid(lower_bound):
                branches.append(new_branch)

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
        print (len(branches))
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















