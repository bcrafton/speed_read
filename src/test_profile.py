
import numpy as np

x = np.load('profile_adc.npy', allow_pickle=True).item()

print (x['wl'])

adc = x[0]['adc']
row = x[0]['row']

weight = np.arange(65, dtype=np.float32)
nrow = np.sum(row * weight, axis=2)

# print (nrow[0])

################################
'''
for i in range(8):
    for j in range(8):
        print (i, j, adc[i][j][16])
'''
'''
for i in range(8):
    for j in range(16):
        print (i, j, nrow[i][j])
'''
'''
for i in range(8):
    for j in range(16):
        print (i, j, row[i][j])
'''
################################

w = np.load('../cifar10_weights.npy', allow_pickle=True).item()
w = w[0]['f']

# print (np.shape(w))
# print (np.max(w))

################################
