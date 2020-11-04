
import numpy as np
import matplotlib.pyplot as plt

###############################

# TRY DIFFERENT LAYERS AS WELL AS [XB, WB, RPR]

###############################

profile = np.load('../profile_adc.npy', allow_pickle=True).item()

adc_count = profile[1]['adc']
row_count = profile[1]['row']

###############################

plt.cla()

y = profile[1]['adc'][0][0][16][0:17]
y = y / np.sum(y)
x = range(len(y))

plt.bar(x=x, height=y, width=0.5)
plt.ylim(0., 0.175)
plt.yticks([0.05, 0.10, 0.15], ['', '', ''])
plt.xticks(x, len(x) * [''])
fig = plt.gcf()
fig.set_size_inches(3.3, 0.8)
plt.tight_layout(0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('pmf1.png', dpi=500)

###############################

plt.cla()

y = profile[1]['adc'][4][0][16][0:17]
y = y / np.sum(y)
x = range(len(y))

plt.bar(x=x, height=y, width=0.5)
plt.ylim(0., 0.175)
plt.yticks([0.05, 0.10, 0.15], ['', '', ''])
plt.xticks(x, len(x) * [''])
fig = plt.gcf()
fig.set_size_inches(3.3, 0.8)
plt.tight_layout(0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('pmf2.png', dpi=500)

###############################











