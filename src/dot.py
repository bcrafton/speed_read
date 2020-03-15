
import numpy as np
from conv_utils import conv_output_length

from scipy.stats import norm, binom
    
##################################################
    
def conv(x, f, b, q, stride, pad1, pad2, params):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co, bpw = np.shape(f)
    assert (bpw == params['bpw'])
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co, params['bpw']))
    y = np.zeros(shape=(Ho, Wo, Co))
    psum = 0

    for h in range(Ho):        
        for w in range(Wo):
            # print ("(%d, %d)" % (h, w))
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :], p = dot(patch, f_matrix, b, q, params)
            psum += p
            # print (y[h, w, :])
            
    return y, psum

def dot(x, w, b, q, params):
    y, psum = pim_dot(x, w, params)
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y, psum
            
##################################################

def pim_dot(x, w, params):
    nrow, ncol, nbit = np.shape(w)
    y = np.zeros(shape=ncol)
    psum = 0
    
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        # print ("%d | %d/%d" % (b, np.sum(xb), np.shape(xb)[0]))

        for r1 in range(0, len(xb), params['wl']):
            r2 = min(r1 + params['wl'], len(xb))
            xbr = xb[r1:r2]
            wr = w[r1:r2]
        
            for c1 in range(0, ncol, params['bl']):
                c2 = min(c1 + params['bl'], ncol)
                wrc = wr[:, c1:c2]
                    
                for b1 in range(0, params['bpw'], 1):
                    b2 = min(b1 + 1, params['bpw'])
                    wrcb = wr[:, :, b1:b2]
            
                    pim, p = pim_kernel(xbr, wrcb, b, b1, params)
                    y[c1:c2] += np.left_shift(pim.astype(int), b)
                    psum += p
            
    return y, psum

##################################################
# subtract 8 here ? 
'''
def sat_err(p, var, adc, rpr, sat):
    def sat_err_help(e, p, var, adc, rpr):
        psum = 0
        for s in range(adc, rpr + 1):
            bin = binom.pmf(s, rpr, p)
            psum += ((s + e) < adc) * bin * (norm.cdf(e + 0.5, 0, var * np.sqrt(s)) - norm.cdf(e - 0.5, 0, var * np.sqrt(s)))
            psum += ((s + e) == adc) * bin * (1 - norm.cdf(adc - s - 0.5, 0, var * np.sqrt(s)))

        # zero case:
        psum += ((e - 0.5 < 0) * (0 < e + 0.5)) * binom.pmf(0, rpr, p)
        return psum
    
    s = np.array(range(-rpr, rpr+1)).reshape(-1, 1)
    pe = sat_err_help(s, p, var, adc, rpr)
    mu = np.sum(pe * s, axis=0)
    mu = mu * sat
    return mu
'''
##################################################

def sat_err(p, var, adc, rpr, sat):
    s = np.array(range(adc, rpr+1)).reshape(-1, 1)
    bin = binom.pmf(s, rpr, p)
    mu = bin * (adc - s)
    mu = np.sum(mu, axis=0)
    return mu

##################################################
'''
def pim_kernel(x, w, xb, wb, params):
    y = 0    
    psum = 0
    
    ishape, oshape, bpw = np.shape(w)
    w_matrix = np.reshape(w, (ishape, oshape * bpw))

    shift = 2 ** np.array(range(wb, wb + bpw))

    wl_ptr = 0
    wl = np.zeros(len(x))
    wl_sum = np.zeros(len(x))
    wl_stride = np.zeros(len(x))
    
    sat = 0
    pdot_sum = 0
    flag = False
    while wl_ptr < len(x):
        # advice = be careful about: (< vs <=), (> vs >=)
        wl[0] = x[0] & (wl_ptr <= 0)
        wl_sum[0] = x[0] & (wl_ptr <= 0)
        wl_stride[0] = (wl_sum[0] <= params['adc'])
        
        for ii in range(1, len(x)):
            if params['skip']:
                row = params['adc'] if flag else params['rpr'][(xb, wb)]
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (wl_sum[ii - 1] < row)
                wl_sum[ii]    = (x[ii] & (wl_ptr <= ii)) + wl_sum[ii - 1]
                wl_stride[ii] = (wl_sum[ii] <= row) + wl_stride[ii - 1]
            else:
                assert (params['rpr'][(xb, wb)] == params['adc'])
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (ii < (wl_ptr + params['adc']))
                wl_stride[ii] = wl_ptr + params['adc']

        pdot = wl @ w_matrix
        assert (np.all(pdot >= 0))
                
        var = np.random.normal(loc=0., scale=params['sigma'] * np.sqrt(pdot), size=np.shape(pdot))
        # TODO: this is wrong i believe.
        # var = var.astype(int)
        var = np.around(var)
        
        pdot = pdot + var
        pdot = np.clip(pdot, 0, params['adc'])
        
        pdot_sum += pdot
        sat += (pdot == params['adc'])

        flag = params['stall'] and (not flag) and (params['rpr'][(xb, wb)] > params['adc']) and (np.any(pdot == params['adc']))
        if not flag:
            wl_ptr = wl_stride[-1]
            y += pdot.reshape(oshape, bpw) @ shift
            if wb == 0:
                x_offset = np.sum(wl).astype(int) * params['offset']
                y -= x_offset

        psum += 1

        p = pdot_sum / (psum * params['rpr'][(xb, wb)])
        mu = np.around(sat_err(p, params['sigma'], params['adc'], params['rpr'][(xb, wb)], sat))
        # assert (np.all(np.absolute(mu) <= 0))
                
        y -= mu.reshape(oshape, bpw) @ shift

    return y, psum
'''
##################################################

def pim_kernel(x, w, xb, wb, params):
    y = 0    
    psum = 0
    ishape, oshape, bpw = np.shape(w)
    w_matrix = np.reshape(w, (ishape, oshape * bpw))

    shift = 2 ** np.array(range(wb, wb + bpw))

    sat = 0
    wl_ptr = 0
    pdot_sum = 0
    wl_sums = 0
    while wl_ptr < len(x):
        wl_sum = 0
        pdot = np.zeros(params['bl'])
        while (wl_ptr < len(x)) and (wl_sum + x[wl_ptr] <= params['rpr'][(xb, wb)]):
            if (x[wl_ptr]):
                wl_sum += 1
                pdot += w_matrix[wl_ptr]

            wl_ptr += 1
                
        var = np.random.normal(loc=0., scale=params['sigma'] * np.sqrt(pdot), size=np.shape(pdot))
        var = np.around(var)
        pdot = pdot + var
        pdot = np.clip(pdot, 0, params['adc'])
        
        pdot_sum += pdot
        sat += (pdot == params['adc'])

        y += pdot.reshape(oshape, bpw) @ shift
        if wb == 0:
            x_offset = wl_sum * params['offset']
            y -= x_offset

        psum += 1
        wl_sums += wl_sum

        # okay we figured out the problem here.
        # using (psum * params['rpr'][(xb, wb)])
        # is bad because brings down the average a lot.

        p = pdot_sum / (psum * params['rpr'][(xb, wb)])
        mu = np.around(sat_err(p, params['sigma'], params['adc'], params['rpr'][(xb, wb)], sat))                
        y -= mu.reshape(oshape, bpw) @ shift
    '''
    if wl_sums > 0:
        p = pdot_sum / wl_sums
        mu = sat * np.around(sat_err(p, params['sigma'], params['adc'], params['rpr'][(xb, wb)], sat))
        mu = mu.reshape(oshape, bpw) @ shift
        y -= mu
    '''

    return y, psum

