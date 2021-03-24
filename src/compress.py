
import numpy as np

x = np.load('profile_adc.npy', allow_pickle=True).item()
y = {}
y['wl']      = x['wl']
y['max_rpr'] = x['max_rpr']

for l in x.keys():
    if (l in ['wl', 'max_rpr']): continue
    # print(np.max(x[l]['adc']))
    y[l] = {}
    y[l]['row']   = x[l]['row']
    y[l]['ratio'] = x[l]['ratio']
    y[l]['adc']   = {}
    for xb in range(8):
        y[l]['adc'][xb] = {}
        for wb in range(8):
            y[l]['adc'][xb][wb] = {}
            for rpr in range(1, 64 + 1):
                hist = x[l]['adc'][xb][wb][rpr][0:(rpr + 1), 0:(rpr + 1)]
                pmf = hist / np.sum(hist)
                y[l]['adc'][xb][wb][rpr] = pmf.astype(np.float16)

np.savez_compressed('y', y)

