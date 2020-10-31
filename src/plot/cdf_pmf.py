
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

###############################
'''
def expected_error(params, adc_count, row_count, rpr, nrow, bias):

    #######################
    # error from rpr <= adc
    #######################
    
    s  = np.arange(rpr + 1, dtype=np.float32)
    
    adc      = np.arange(params['adc'] + 1, dtype=np.float32).reshape(-1, 1)
    adc_low  = np.array([-1e6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
    adc_high = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)
    
    pe = norm.cdf(adc_high, s, params['sigma'] * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, params['sigma'] * np.sqrt(s) + 1e-6)
    e = s - adc
    p = adc_count[rpr, 0:rpr + 1] / (np.sum(adc_count[rpr]) + 1e-6)

    #######################
    # error from rpr > adc
    #######################
    
    if rpr > params['adc']:
        e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - bias
        # e[:, params['adc']:rpr+1] = e[:, params['adc']:rpr+1] - round(nrow * bias) // nrow

    # mse = np.sum((p * pe * e * nrow) ** 2)
    # mse = np.sqrt(np.sum((p * pe * e * nrow) ** 2))
    # mse = np.sqrt(np.sum((p * pe * e) ** 2) * nrow)
    mse = np.sum(np.absolute(p * pe * e * nrow))

    mean = np.sum(p * pe * e * nrow)

    return mse, mean
'''
###############################

s  = np.arange(8 + 1, dtype=np.float32)

adc      = np.arange(8 + 1, dtype=np.float32).reshape(-1, 1)
adc_low  = np.array([-1e6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]).reshape(-1, 1)
adc_high = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1e6]).reshape(-1, 1)

pe = norm.cdf(adc_high, s, 0.15 * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, 0.15 * np.sqrt(s) + 1e-6)

# plt.imshow(1 - pe, cmap='hot', interpolation='nearest')
plt.imshow(1 - pe, cmap='gray', interpolation='nearest')
# plt.show()

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

for i in range(0, 8+1):
  for j in range(0, 8+1):
      if pe[i, j] < 0.01:
          pass
      elif pe[i, j] < 0.5:
          text = plt.text(j, i, '%0.2f' % pe[i, j], ha="center", va="center", color="black")
      else:
          text = plt.text(j, i, '%0.2f' % pe[i, j], ha="center", va="center", color="white")

# plt.show()
# plt.imsave('adc_pmf.png', dpi=300)
plt.savefig('adc_pmf.png', dpi=300)

###############################













