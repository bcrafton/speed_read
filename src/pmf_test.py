
import numpy as np
import matplotlib.pyplot as plt

pmf = np.load('profile.npz', allow_pickle=True)['arr_0'].item()
# print (pmf.keys())

'''
x = pmf[0]['adc']
# print (x.keys())
# print (x[7].keys())
x = x[7][0][16][0:16+1][0:16+1]
# print (np.sum(x))

plt.imshow(x.astype(np.float32))
plt.show()
'''

x = pmf[3]['adc']
x = x[7][0][16][0:16+1][0:16+1]
print (x)
plt.imshow(x.astype(np.float32))
plt.show()


