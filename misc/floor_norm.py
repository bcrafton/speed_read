
import numpy as np
import matplotlib.pyplot as plt

xs = np.array(range(10))
ys = []

for x in xs:
    y = np.random.normal(loc=0., scale=x, size=100000)
    y = np.clip(y, 0., 1e6)
    y = np.floor(y)
    y = np.std(y)
    ys.append(y)
    
plt.plot(xs, ys)

m = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
ys = m * xs
plt.plot(xs, ys)

print (m)

plt.show()



