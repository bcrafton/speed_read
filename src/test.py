
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
scale = 2 ** np.arange(8)
scale = scale.reshape(-1,1,1,1,1) * scale.reshape(1,-1,1,1,1)
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
    error = 0
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
            output = np.random.choice(a=a, size=1, replace=True, p=p)
            # check if value != e
            ADC = value[xb][wb][output]
            error += 2 ** (xb + wb) * (ADC - ON) / q * ratio
    errors.append(error)

mean = np.mean(errors)
mae = np.mean(np.abs(errors - mean))
print (mae, mean, np.std(errors))

####################################################

# errors = np.array(errors) / np.mean(np.abs(errors))
# plt.hist(errors, bins=100)
# plt.show()
# plt.savefig('img.png', dpi=300)

####################################################

p = np.reshape(profile, (8, 8, 65, 65, 1))
pe = np.reshape(conf, (8, 8, 65, 65, 65))
e = value.reshape(8, 8, 1, 1, 65) - np.arange(65).reshape(1, 1, 1, 65, 1)

PE = np.sum(p * pe * (np.abs(e) > 0), axis=(2, 3, 4))
print (np.around(PE, 3))

E = np.sum(p * pe * ratio / q * (np.abs(e) * scale) ** 2, axis=(2, 3, 4))
print (np.around(E, 3))

PE = PE * (E / np.max(E))
print (np.around(PE, 3))

print (np.sum(PE))

####################################################






