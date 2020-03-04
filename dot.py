
import numpy as np

##################################################

def pim_dot(x, w, bpa, bpw, rpr):
    y = 0
    for b in range(bpa):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        pim = pim_dot_kernel(xb, w, bpa, bpw, rpr)
        y += np.left_shift(pim.astype(int), b)
        
    return y
            
##################################################

def pim_dot_kernel(x, w, bpa, bpw, rpr):
    wl_ptr = 0
    wl = np.zeros(128)
    wl_sum = np.zeros(128)
    wl_stride = np.zeros(128)
    
    y = 0
    while wl_ptr < 128:
        wl[0] = x[0] & (wl_ptr <= 0)
        wl_sum[0] = x[0] & (wl_ptr <= 0)
        wl_stride[0] = (wl_sum[0] <= 8)
        
        for ii in range(1, 128):
            wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (wl_sum[ii - 1] < 8)
            wl_sum[ii]    = (x[ii] & (wl_ptr <= ii)) + wl_sum[ii - 1]
            wl_stride[ii] = (wl_sum[ii] <= 8) + wl_stride[ii - 1]

        wl_ptr = wl_stride[-1]
        y += wl @ w
        
    return y
    
##################################################

x = np.random.choice(a=np.array(range(256)), size=128, replace=True).astype(int)
w = np.random.choice(a=np.array(range(256)), size=(128, 128), replace=True).astype(int)
y = pim_dot(x, w, 8, 4, 8)

print (np.all(y == (x @ w)))

##################################################



