
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt 

on = 5
density = 0.5
rpr = 10
b = binom.pmf(on, rpr, density)
print (b)

on = np.array(range(10 + 1))
b = binom.pmf(on, rpr, density)
plt.plot(on, b)
plt.show()


