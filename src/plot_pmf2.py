
import numpy as np
import matplotlib.pyplot as plt

###############################

profile = np.load('profile/1.npy', allow_pickle=True).item()
profile = profile['adc']
################################################################################## 
profile = np.sum(profile[0, 0, 64, :, :], axis=0)
# profile = profile[0, 0, 64, 64, :]
##################################################################################
profile = profile / np.sum(profile)
profile = np.around(profile, 3)
print (profile)

codes = np.arange(len(profile))
print (codes)

ymax = np.max(profile) * 1.1

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

thresh = thresh[(0, 0, 0, 8, 0, 2)]
# print (thresh)

plt.vlines(x=thresh, ymin=0, ymax=ymax, colors='black')

###############################

plt.bar(x=codes, height=profile, width=0.5)
plt.ylim(0., ymax)
# plt.yticks([0.25, 0.50, 0.75], ['', '', ''])
plt.xticks(codes, len(codes) * [''])
fig = plt.gcf()
# fig.set_size_inches(3.3, 0.8)
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('adc_pmf.png', dpi=500)

###############################
