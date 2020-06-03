
import numpy as np

cycle = np.zeros(shape=(2, 2, 2, 2, 10))
cycle[0][0][0][0][5] = 1

print (np.where(cycle == 1))

idx = tuple([[0], [0], [0], [0], [5]])
print (cycle[idx])


