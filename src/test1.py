
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

####################################################
dump = np.load('dump.npy', allow_pickle=True).item()
####################################################
conf = dump['conf']
conf = conf / np.sum(conf, axis=-1, keepdims=True)
value = dump['value']
profile = dump['profile']
profile = profile / np.sum(profile, axis=(2, 3), keepdims=True)
ratio = dump['ratio']
q = dump['q']
####################################################
'''
print (np.shape(profile))
print (np.shape(conf))
print (np.shape(value))
'''
####################################################
profile = np.reshape(profile, (8, 8, 65 * 65))
conf = np.reshape(conf, (8, 8, 65 * 65, 65))
####################################################
errors = []
for itr in range(1000):
    print (itr)
    error = []
    for xb in range(8):
        for wb in range(8):
            # probability of [WL, ON]
            a = np.arange(65 * 65, dtype=int)
            p = profile[xb][wb]
            input = np.random.choice(a=a, size=1, replace=True, p=p).item()
            WL = input // 65
            ON = input % 65
            # probability of E given [WL, ON]
            a = np.arange(65, dtype=int)
            p = conf[xb][wb][input]
            output = np.random.choice(a=a, size=1, replace=True, p=p).item()
            # check if value != e
            ADC = value[xb][wb][output]
            e = 2 ** (xb + wb) * (ADC - ON) / q * ratio
            # e = (ADC - ON)
            if abs(e) > 0: error.append(e)
    errors.append(error)
    print (error)

total = 0
for error in errors:
    total += len(error)
    

'''
mean = np.mean(errors)
mae = np.mean(np.abs(errors - mean))
print (mae, mean)
'''

'''
errors = np.array(errors) / np.mean(np.abs(errors))
plt.hist(errors, bins=100)
plt.show()
'''


####################################################





