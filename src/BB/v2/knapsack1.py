
import numpy as np
import copy

############################

capacity = 165
w = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82], dtype=np.float)
p = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72], dtype=np.float)
N = len(w)
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
            if total_weight + w[i] < capacity:
                total_weight += w[i]
                total_value += p[i]

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
            if total_weight + w[i] < capacity:
                total_weight += w[i]
                total_value += p[i]
            else:
                remainder = capacity - total_weight
                total_value += p[i] * (remainder / w[i])
                total_weight += remainder

        return total_value

    def branch(self, lower_bound):
        return branches

############################

def branch_and_bound_help(branches, lower_bound):
    return new_branches

def branch_and_bound():
    return branches
        
############################

branches = branch_and_bound()

############################



















