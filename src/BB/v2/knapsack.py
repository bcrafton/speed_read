
import numpy as np

capacity = 165
w = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82], dtype=np.float)
p = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72], dtype=np.float)
N = len(w)
truth = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

ratio = p / w
sort = np.argsort(ratio)
w = w[sort]
p = w[sort]

total_weight = 0
total_profit = 0
for i in range(N):
    if total_weight + w[i] < capacity:
        total_weight += w[i]
        total_profit += p[i]

print (total_profit)
