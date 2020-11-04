
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
import matplotlib

###############################

s  = np.arange(8 + 1, dtype=np.float32)

adc      = np.arange(8 + 1, dtype=np.float32).reshape(-1, 1)
adc_low  = np.array([-1e6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
adc_high = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)

pe = norm.cdf(adc_high, s, 0.15 * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, 0.15 * np.sqrt(s) + 1e-6)
pe = pe * 100

plot = pe
plot = np.where(plot >= 0.1, np.log(plot), np.zeros_like(plot))
plot = plot / np.sum(plot, axis=1)
plot = plot * 100
plot = 175 - plot

###############################

normalize = matplotlib.colors.Normalize(vmin=0, vmax=200)
plt.imshow(plot, cmap='gray', interpolation='nearest', norm=normalize)

for i in range(0, 8+1):
  for j in range(0, 8+1):
      if pe[i, j] > 99.95:
          text = plt.text(j, i, '100', ha="center", va="center", color="black")
      else:
          text = plt.text(j, i, '%0.1f' % pe[i, j], ha="center", va="center", color="black")

# plt.show()
# plt.imsave('adc_pmf.png', dpi=300)
plt.savefig('adc_pmf.png', dpi=300)

###############################













