
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# not sure how fast we can go with this.
# will need 16 luts for rpr=16.
# each will be 1000 entries.

# think we shud see how 

std = 0.1 * np.sqrt(8)

print( norm.cdf(2.5, 0, std) - norm.cdf(1.5, 0, std) )
print( norm.cdf(1.5, 0, std) - norm.cdf(0.5, 0, std) )
print( norm.cdf(0.5, 0, std) - norm.cdf(-0.5, 0, std) )
print( norm.cdf(-0.5, 0, std) - norm.cdf(-1.5, 0, std) )
print( norm.cdf(-1.5, 0, std) - norm.cdf(-2.5, 0, std) )
