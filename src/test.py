
import numpy as np

####################################################
dump = np.load('dump.npy', allow_pickle=True).item()
####################################################
conf = dump['conf']
conf = conf / np.sum(conf, axis=-1, keepdims=True)
value = dump['value']
profile = dump['profile']
profile = profile / np.sum(profile, axis=(2, 3), keepdims=True)
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
xb = 0
wb = 2
####################################################
errors = 0
samples = 10000
for _ in range(10000):
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
    errors += (ADC != ON)
####################################################
print (errors / samples)
####################################################

p = profile[xb][wb] 
pe = conf[xb][wb]
e = 1 - (np.arange(65, dtype=int).reshape(-1, 1) == value[xb][wb])

# WL, ON
print (np.shape(p))
# WL, ON, ADC
print (np.shape(pe))
# ON, ADC
print (np.shape(e))

p  = np.reshape(p,  (65, 65,  1))
pe = np.reshape(pe, (65, 65, 65))
e  = np.reshape(e,  ( 1, 65, 65))

print (np.sum(p * pe * e))























