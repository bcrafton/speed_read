
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

# def patches(x, fh, fw, s, p1, p2, wl, bpa):
x = patches(x2, 7, 7, 2, 3, 3, 128, 8)
# print (np.shape(x))
npatch, nwl, wl, bpa = np.shape(x)
x = np.transpose(x, (0,3,1,2))
# print (np.shape(x))
x = np.reshape(x, (npatch * bpa, nwl, wl))
# print (np.shape(x))

#########################

w = cut(w2, 8, 128, 128)
nwl, wl, nbl, bl = np.shape(w)

print (np.shape(x), np.std(x))
print (np.shape(w), np.std(w))

#########################

psums = [] 

for p in range(npatch):
    for i in range(nwl):
        wlsum = 0
        psum = np.zeros(shape=(nbl, bl))
        
        for j in range(wl):
        
            if x[p][i][j]:
                wlsum += 1
                psum += w[i][j]
                
            if wlsum == RPR:
                wlsum = 0
                psums.append(psum)
                psum = np.zeros(shape=(nbl, bl))
        
        if wlsum < RPR:
            psums.append(psum)

#########################

psums = np.array(psums)
psums = np.reshape(psums, (-1, 1))

values, counts = np.unique(psums, return_counts=True)
# plt.hist(x)
# plt.show()
print (values)
print (counts)

#########################

# kmeans = KMeans(n_clusters=ADC, init='k-means++', max_iter=1000, n_init=50, random_state=0)
# kmeans.fit(psums)

# centroids = np.round(kmeans.cluster_centers_[:, 0], 2)
# print (centroids)

#########################

np.save('psums', psums)







