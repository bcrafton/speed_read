
import numpy as np
import matplotlib.pyplot as plt

X = 4
W = 4
RPR = 32
###############################
profile = np.load('profile/1.npy', allow_pickle=True).item()
profile = profile['adc']
################################################################################## 
profile = np.sum(profile[X, W, RPR, :, :], axis=0)[0:RPR + 1]
# profile = profile[X, W, 64, 64, :]
##################################################################################
profile = profile / np.sum(profile)
profile = np.around(profile, 3)
print (profile)

codes = np.arange(len(profile))
print (codes)

# ymax = np.max(profile) * 1.1
ymax = 0.15

xmin = np.min(codes)
xmax = np.max(codes)

###############################

kmeans = np.load('kmeans_1.npy', allow_pickle=True).item()
thresh, value, delay, error = kmeans['thresh'], kmeans['value'], kmeans['delay'], kmeans['error']

# print (thresh.keys())
# 
# thresh_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)] = thresh
# value_table[(xb, wb, step_idx, rpr_idx, adc_idx, sar_idx)] = values
# 
# STEP = np.array([1]) - 1
# RPR = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]) - 1
# ADC = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64]) - 1
# SAR = np.array([0, 2, 3, 4, 5, 6])
# 
# (8, 0, 3) = 64 RPR, 1 ADC, 4 SAR

thresh = thresh[(X, W, 0, 6, 0, 3)]
print (thresh)

plt.vlines(x=thresh, ymin=0, ymax=ymax, colors='black', linewidth=0.5, linestyle='dashed')

###############################

plt.bar(x=codes, height=profile, width=0.5)

plt.ylim(0., ymax)
plt.yticks([0.05, 0.10, 0.15], ['', '', ''])

# plt.xticks(codes, len(codes) * [''])

plt.xlim(xmin - 0.75, xmax + 0.75) # (-0.75) bc (-1) -> adds a tick for -1. 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.gca().xaxis.set_major_locator(MultipleLocator(8))
plt.gca().xaxis.set_major_formatter('')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_formatter('')

fig = plt.gcf()
fig.set_size_inches(3.3, 0.8)
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('adc_pmf2a.png', dpi=500, transparent=True)

###############################






