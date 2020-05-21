
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#########################

RPR = 12
ADC = 8

#########################

x = np.load('../src/resnet18_activations.npy', allow_pickle=True).item()
# print (x.keys())
# x2 = x[0][0]
x2 = x['x'][0]
# print (np.shape(x2))

w = np.load('../src/resnet18_quant_weights.npy', allow_pickle=True).item()

# print (w.keys())
# print (w[3].keys())
w2 = w[0]['f']

#########################

def patches(x, fh, fw, s, p1, p2, wl, bpa):
    
    xh, xw, xc = np.shape(x)
    
    yh = (xh - fh + s + p1 + p2) // s
    yw = yh

    #########################
    
    x = np.pad(array=x, pad_width=[[p1,p2], [p1,p2], [0,0]], mode='constant')
    patches = []
    for h in range(yh):
        for w in range(yw):
            patch = np.reshape(x[h*s:(h*s+fh), w*s:(w*s+fw), :], -1)
            patches.append(patch)
            
    #########################
    
    patches = np.stack(patches, axis=0)
    pb = []
    for xb in range(bpa):
        pb.append(np.bitwise_and(np.right_shift(patches.astype(int), xb), 1))
    
    patches = np.stack(pb, axis=-1)
    npatch, nrow, nbit = np.shape(patches)
    
    #########################
    
    if (nrow % wl):
        zeros = np.zeros(shape=(npatch, wl - (nrow % wl), bpa))
        patches = np.concatenate((patches, zeros), axis=1)
        
    patches = np.reshape(patches, (npatch, -1, wl, bpa))
    
    #########################
    
    return patches
    
#########################

def cut(w, bpw, wl, bl):

    fh, fw, fc, fn = np.shape(w)

    w_offset = w + 128
    w_matrix = np.reshape(w_offset, (fh * fw * fc, fn))
    wb = []
    for bit in range(bpw):
        wb.append(np.bitwise_and(np.right_shift(w_matrix, bit), 1))
    wb = np.stack(wb, axis=-1)

    ########################

    nrow, ncol, nbit = np.shape(wb)
    if (nrow % wl):
        zeros = np.zeros(shape=(wl - (nrow % wl), ncol, nbit))
        wb = np.concatenate((wb, zeros), axis=0)

    nrow, ncol, nbit = np.shape(wb)
    wb = np.reshape(wb, (-1, wl, ncol, nbit))

    ########################

    nwl, wl, ncol, nbit = np.shape(wb)
    # wb = np.transpose(wb, (0, 1, 3, 2))
    wb = np.reshape(wb, (nwl, wl, ncol * nbit))

    nwl, wl, ncol = np.shape(wb)
    if (ncol % bl):
        zeros = np.zeros(shape=(nwl, wl, bl - (ncol % bl)))
        wb = np.concatenate((wb, zeros), axis=2)

    wb = np.reshape(wb, (nwl, wl, -1, bl))
    
    ########################
    
    return wb

#########################

x = patches(x2, 7, 7, 2, 3, 3, 128, 8)
npatch, nwl, wl, bpa = np.shape(x)

#########################

w = cut(w2, 8, 128, 128)
nwl, wl, nbl, bl = np.shape(w)

# print (np.shape(x))
# (12544, 2, 128, 8)

# print (np.shape(w))
# (2, 128, 4, 128)

#########################

# (12544, 2, 128, 8)
x = np.mean(x, axis=(0, 3))
# (2, 128)
x = np.reshape(x, (2, 128, 1, 1))

#########################

p = x * w
# print (np.shape(p))
# (2, 128, 4, 128)

p = np.transpose(p, (0,2,3,1))
p = np.reshape(p, (nwl * nbl * bl, wl))
# print (np.shape(p))
# (1024, 128)

#########################

def prob(ps):
    N, P = np.shape(ps)
    dist = np.zeros(shape=(N, P + 1))
    dist[:, 0] = 1.
    for p in range(P):
        new_dist = np.copy(dist)
        new_dist[:, 0:P]    = dist[:, 0:P] * (1 - ps[:, p].reshape(N, 1))
        new_dist[:, 1:P+1] += dist[:, 0:P] * ps[:, p].reshape(N, 1)
        dist = new_dist

    return dist

############################

out = prob(p)
print (np.shape(out))

############################
























