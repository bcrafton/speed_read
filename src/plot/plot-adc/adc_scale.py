
import numpy as np
import matplotlib.pyplot as plt

###############################

def adc(x, N):
    ret = 0
    for a in range(N):
        ret += (x > (a - 0.5)) * (x < (a + 0.5)) * a
    
    ret += (x > (N - 0.5)) * N
    return ret

###############################

def adc_scaled(x, N):
    x = x / 2
    
    ret = 0
    for a in range(N):
        ret += (x > (a - 0.5)) * (x < (a + 0.5)) * a
    
    ret += (x > (N - 0.5)) * N
    return ret

###############################

def adc_shift(x, N):
    x = x - 3

    ret = 0
    for a in range(N):
        ret += (x > (a - 0.5)) * (x < (a + 0.5)) * a
    
    ret += (x > (N - 0.5)) * N
    return ret

###############################

x = np.linspace(0, 16, 1000)    

y1 = adc(x, 8)
y2 = adc_scaled(x, 8)
y3 = adc_shift(x, 8)

###############################

plt.cla()
plt.plot(x, y1, color='black')
plt.plot(x, y2, color='blue')

ax = plt.gca()
fig = plt.gcf()
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='dotted')
fig.set_size_inches(4., 3.)
plt.tight_layout()
fig.savefig('adc_scale', dpi=300)
# plt.show()

###############################

plt.cla()
plt.plot(x, y1, color='black')
plt.plot(x, y3, color='blue')

ax = plt.gca()
fig = plt.gcf()
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='dotted')
fig.set_size_inches(4., 3.)
plt.tight_layout()
fig.savefig('adc_shift', dpi=300)
# plt.show()







