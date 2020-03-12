import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

# Fix parameters:
Nrpr = 10
Nadc = 8
r = 8
std = 0.2
p = 0.5
number_reads = 16

def p0_sum_rpr(a):
    p0_of_a = 0
    # Use state variables Nrpr, p:
    for s in range(1, Nrpr + 1):
        bin = binom.pmf(s, Nrpr, p)
        c = s + a
        if (c < Nadc):  p0_of_a += bin * (norm.cdf((a + 0.5) / (std * np.sqrt(s))) - norm.cdf((a - 0.5) / (std*np.sqrt(s))))
        if (c == Nadc): p0_of_a += bin * (1 - norm.cdf((Nadc - s - 0.5) / (std * np.sqrt(s))))

    # The PDF for normal becomes dirac delta when s = 0, so prob. over some range is 1 as long as range includes 0:
    if (a - 0.5 < 0 < a + 0.5): p0_of_a += binom.pmf(0, Nrpr, p)

    return p0_of_a

# Normal Approximation:
sigma_hat = 0
for a in range(-r, r+1):
    sigma_hat += p0_sum_rpr(a) * (a ** 2)

###############

sigma_tot = np.sqrt(sigma_hat * number_reads)
print (sigma_tot, sigma_hat)

x = np.arange(-number_reads*r, number_reads*r + 1)

p = norm.pdf(x, 0, sigma_tot)     
plt.plot(x,p)
plt.show()
