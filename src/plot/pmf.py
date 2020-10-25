
import numpy as np
import matplotlib.pyplot as plt

###############################

# TRY DIFFERENT LAYERS AS WELL AS [XB, WB, RPR]

###############################

profile = np.load('../profile_adc.npy', allow_pickle=True).item()

adc_count = profile[1]['adc']
row_count = profile[1]['row']

# print (np.shape(adc_count))
# print (np.shape(row_count))

###############################

def xy(x, w, rpr=16):
    y = adc_count[x][w][rpr][0:rpr+1]
    y = y / np.sum(y)
    x = range(len(y))
    return x, y
    
###############################

'''
fig, axs = plt.subplots(3, 1)

x, y = xy(0,0,16)
axs[0].bar(x=x, height=y, width=0.5)
axs[0].set_xticks(x)

x, y = xy(6,0,16)
axs[1].bar(x=x, height=y, width=0.5)
axs[1].set_xticks(x)

x, y = xy(5,5,16)
axs[2].bar(x=x, height=y, width=0.5)
axs[2].set_xticks(x)
'''

###############################

fig, axs = plt.subplots(2, 1)

x, y = xy(0,0,16)
axs[0].bar(x=x, height=y, width=0.5)
axs[0].set_xticks(x)

x, y = xy(4,0,16)
axs[1].bar(x=x, height=y, width=0.5)
axs[1].set_xticks(x)

###############################

fig = plt.gcf()
fig.set_size_inches(5., 4.)
plt.tight_layout()
plt.savefig('pmf.png', dpi=300)

###############################











