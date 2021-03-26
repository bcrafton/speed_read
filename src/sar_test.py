
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 64, 1000)

def sar_step(x, adc, step):
    y = np.minimum(x, adc)
    y = np.around(y // 4) * 4
    return y

y = sar_step(x, 64, 1000)
plt.plot(x, y)
y = sar_step(x, 16, 1000)
plt.plot(x, y)

plt.show()

