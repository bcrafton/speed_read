
import numpy as np
import matplotlib.pyplot as plt 

acc = np.load('cifar_acc.npy', allow_pickle=True).item()
sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(0, 0, 'dynamic', sigma, 0.1)]['acc'] #* 100.
    ys.append(y)
plt.plot(xs, ys, marker='.', color='green', markersize=3, linewidth=1)

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 0, 'dynamic', sigma, 0.1)]['acc'] #* 100.
    ys.append(y)
plt.plot(xs, ys, marker='.', color='blue', markersize=3, linewidth=1)

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 0.1)]['acc'] #* 100.
    ys.append(y)
plt.plot(xs, ys, marker='.', color='#808080', markersize=3, linewidth=1)

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 0.5)]['acc'] #* 100.
    ys.append(y)
plt.plot(xs, ys, marker='.', color='#404040', markersize=3, linewidth=1)

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 1.0)]['acc'] #* 100.
    ys.append(y)
plt.plot(xs, ys, marker='.', color='#000000', markersize=3, linewidth=1)

###############################

plt.ylim(bottom=0, top=105.)
plt.yticks([25, 50, 75, 100], ['', '', '', ''])
plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20], ['', '', '', '', ''])

plt.grid(True, linestyle='dotted')
plt.gcf().set_size_inches(3.3, 1.)
plt.tight_layout(0.)
plt.savefig('cc_acc.png', dpi=500)

###############################










