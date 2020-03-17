
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt 

#######################

def factorial(n):
  fact = 1
  for i in range(1, n+1): 
      fact = fact * i 
  return fact
  
def binomial_pmf(k, n, p):
    nCk = factorial(n) / (factorial(k) * factorial(n - k))
    success = p ** k
    fail = (1 - p) ** (n - k)
    return nCk * success * fail

#######################


# print (binom.pmf(10, 16, 0.4))    
# print (binomial_pmf(10, 16, 0.4))

#######################

def nCk(n, k):
    t = factorial(n)
    b = factorial(k) * factorial(n - k)
    print (t, b)
    return t / b

#######################

print (nCk(16, 10))

#######################



