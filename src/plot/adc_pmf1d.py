
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

###############################

plt.cla()

y = 1 - np.sum(pe * np.eye(9), axis=1)
x = range(len(y))

plt.bar(x=x, height=y, width=0.5)
plt.ylim(0., 1.)
plt.yticks([0.25, 0.50, 0.75], ['', '', ''])
plt.xticks(x, len(x) * [''])
fig = plt.gcf()
fig.set_size_inches(3.3, 0.8)
plt.tight_layout(0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('adc_pmf.png', dpi=500)

###############################












