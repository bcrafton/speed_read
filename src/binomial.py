
import numpy as np
from scipy.stats import norm, binom

# https://www.geeksforgeeks.org/binomial-random-variables/
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm
# http://www.cplusplus.com/reference/random/binomial_distribution/

print (binom.pmf(5, 10, 0.5))

def factorial(n):
  fact = 1
  for i in range(1,n+1): 
      fact = fact * i 
  return fact
  
def binomial_pmf(k, n, p):
    nCk = factorial(n) / (factorial(k) * factorial(n - k))
    success = p ** k
    fail = (1 - p) ** (n - k)
    return nCk * success * fail
    
print (binomial_pmf(5, 10, 0.5))
