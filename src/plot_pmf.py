
import numpy as np
import matplotlib.pyplot as plt

###############################

profile = np.load('profile/1.npy', allow_pickle=True).item()
profile = profile['adc']
profile = np.sum(profile[0, 0, 64, :, :], axis=0)
profile = profile / np.sum(profile)
profile = np.around(profile, 3)
print (profile)

codes = np.arange(len(profile))
print (codes)

###############################

kmeans = np.load('kmeans_1.npy', allow_pickle=True).item()
thresh, value, delay, error = kmeans['thresh'], kmeans['value'], kmeans['delay'], kmeans['error']

###############################

plt.bar(x=codes, height=profile, width=0.5)
# plt.ylim(0., 1.)
# plt.yticks([0.25, 0.50, 0.75], ['', '', ''])
plt.xticks(codes, len(codes) * [''])
fig = plt.gcf()
# fig.set_size_inches(3.3, 0.8)
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('adc_pmf.png', dpi=500)

###############################
