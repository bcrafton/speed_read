
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#################################################

lrss = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
sar   = np.load('sar_plot.npy', allow_pickle=True).item()
flash = np.load('flash_plot.npy', allow_pickle=True).item()

#################################################

color = {
(0, 0.10): 'blue',
(0, 0.25): 'blue',
(1, 0.10): '#808080',
(1, 0.25): '#404040',
(1, 0.50): '#000000',
}

######################################

plt.cla()

for key in sar['perf'].keys():
  plt.plot(lrss, sar['perf'][key], color=color[key], marker='.', markersize=3, linewidth=1)

for key in flash['perf'].keys():
  plt.plot(lrss, flash['perf'][key], color=color[key], marker='.', markersize=3, linewidth=1)

plt.ylim(bottom=0, top=32.5)
plt.yticks([0, 10, 20, 30], ['', '', '', ''])
plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12], ['', '', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(1.65, 1.)
plt.tight_layout(0.)
plt.savefig('cc_perf.png', dpi=500)

####################
'''
plt.cla()
for key in sar['power'].keys():
  plt.plot(lrss, sar['power'][key], color=color[key], marker='.', markersize=3, linewidth=1)
plt.ylim(bottom=0, top=15)
plt.yticks([2, 4, 6, 8], ['', '', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])
plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(1.65, 1.)
plt.tight_layout(0.)
plt.savefig('cc_power.png', dpi=500)
'''
#################################################

