
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

colors = ['royalblue', 'gray', 'black']

for sigma in [0.07]:
    for layer in range(6):
        query = '(sigma == %f) & (id == %d)' % (sigma, layer)
        samples = df.query(query)
        ys = samples['error_count'].to_numpy()[0]
        for i, y in enumerate(ys):
            plt.bar(x=layer+1+i*0.25-0.25, height=y, width=0.25, color=colors[i])

plt.yscale('log')
plt.ylim(top=0.5*10**8)
plt.yticks(10**np.array([1,3,5,7]), 4 * [''])

plt.xticks([1,2,3,4,5,6], 6 * [''])

plt.gcf().set_size_inches(3.3, 1.)
plt.grid(True, axis='y', linestyle='dotted') # , color='black'
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('error_rate.png', dpi=300)

######################################















