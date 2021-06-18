
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####################

def ld_to_dl(ld):
    dl = {}

    for i, d in enumerate(ld):
        for key in d.keys():
            value = d[key]
            if i == 0:
                dl[key] = [value]
            else:
                dl[key].append(value)

    return dl

####################

results = np.load('../results.npy', allow_pickle=True)
# print (len(results))

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['rpr_alloc'])
# print (df['example'])

####################
'''
for sigma in [0.07]:
    query = '(sigma == %f)' % (sigma)
    samples = df.query(query)
    print (samples['error_count'])
'''
######################################
'''
colors = ['gray', 'royalblue', 'black']
# colors = ['black', 'green', 'red']

for sigma in [0.07]:
    for layer in range(8):
        query = '(sigma == %f) & (id == %d)' % (sigma, layer)
        samples = df.query(query)
        ys = samples['error_count'].to_numpy()[0]
        for i, y in enumerate(ys):
            plt.bar(x=layer+1, height=y, color=colors[i])
'''
######################################
'''
colors = ['gray', 'royalblue', 'black']

for sigma in [0.07]:
    for layer in range(6):
        query = '(sigma == %f) & (id == %d)' % (sigma, layer)
        samples = df.query(query)
        ys = samples['error_count'].to_numpy()[0]
        for i, y in enumerate(ys):
            plt.bar(x=layer+1+i*0.25-0.25, height=y, width=0.25, color=colors[i])

plt.yscale('log')
plt.savefig('error_rate.png', dpi=500)
'''
######################################

colors = ['#228B22', '#96F078', '#ffae42', '#ff849c', '#ff0000']
layers = np.arange(8, dtype=int)

sigma = 0.07
N = 3
    
for layer in layers:
    query = '(sigma == %f) & (id == %d)' % (sigma, layer)
    samples = df.query(query)
    ys = samples['error_count'].to_numpy()[0]
    
    # N = len(ys)
    width = 0.75 / N
    for i, y in enumerate(ys):
        x = layer + (i+0.5)*width - (N/2)*width 
        plt.bar(x=x, height=y, width=width, color=colors[i])

plt.yscale('log')
plt.ylim(bottom=0.25, top=10**9)
plt.yticks(10**np.array([1,3,5,7]), 4 * [''])

plt.xticks(layers, len(layers) * [''])

plt.gcf().set_size_inches(3.5, 0.8)
plt.grid(True, axis='y', linestyle='dotted', color='black')
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('error_rate_%f.png' % (sigma), dpi=500)

######################################















