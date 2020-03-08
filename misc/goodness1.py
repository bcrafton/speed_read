
import numpy as np
import matplotlib.pyplot as plt

##############################
        
def goodness(x):
    x = np.reshape(x, -1).astype(int)
    
    count = np.zeros(8)
    for bit in range(8):
        count[bit] = np.sum(np.bitwise_and(np.right_shift(x, bit), 1))

    g = count / len(x)
    return np.max(g)

##############################

weights = np.load('cifar10_weights_exp.npy', allow_pickle=True).item()
x = weights[2][0]

##############################

offsets = range(64, 192+1)
good = []
for offset in offsets:
    good.append(goodness(x + offset))

print (np.stack((offsets, good), axis=1))

plt.plot(offsets, good)
plt.show()

##############################











