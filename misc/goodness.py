
import numpy as np
import matplotlib.pyplot as plt

##############################

def count_ones(x):
    count = 0
    for bit in range(8):
        count += np.bitwise_and(np.right_shift(x, bit), 1)
    return count
        
def goodness(x):
    count = count_ones(x)
    return np.sum(count) / np.prod(np.shape(count))

##############################

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()
x = np.reshape(weights[2][0], -1).astype(int)

##############################

print (np.min(x), np.max(x))
print (goodness(x))

##############################

'''
y = x + 128
print (np.min(y), np.max(y))
print(goodness(y))
plt.hist(y, bins=100)
plt.show()
'''

##############################

'''
y = x + 128 + 32 # 32 -> 261 is actually correct because weights are trained for [-128, 127] this works for [-160, 95]
print (np.min(y), np.max(y))
print(goodness(y))
plt.hist(y, bins=100)
plt.show()
'''

##############################

offsets = range(64, 192+1)
good = []
for offset in offsets:
    good.append(goodness(x + offset))

plt.plot(offsets, good)
plt.show()

##############################











