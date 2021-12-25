
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

refs = np.arange(0, 64, 1).astype(int)
plt.vlines(x=refs, ymin=0, ymax=1., colors='gray', linewidth=0.5, linestyle='dashed')

refs = np.arange(0, 8, 1).astype(int)
plt.vlines(x=refs, ymin=0, ymax=1., colors='red', linewidth=0.5)

#################################################
'''
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
y = norm.pdf(x)
plt.plot(x, y)
'''
#################################################

# plt.xticks([])
plt.yticks([])

#################################################

xmin = 0
xmax = 64
plt.xlim(xmin - 0.75, xmax + 0.75) # (-0.75) bc (-1) -> adds a tick for -1. 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.gca().xaxis.set_major_locator(MultipleLocator(8))
plt.gca().xaxis.set_major_formatter('')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_formatter('')

#################################################

fig = plt.gcf()
fig.set_size_inches(3.3, 0.3)
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.grid(True, axis='y', linestyle='dotted')
plt.savefig('ref1.png', dpi=500, transparent=True)

