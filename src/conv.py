
import math
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
from kmeans import kmeans

from conv_utils import *
from cdot import *
from dot_ref import *
from var import *

from layers import *
from cprofile import profile
from rpr import rpr as dynamic_rpr

#########################

def adc_range(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_low = np.zeros_like(adc, dtype=np.float32)
    adc_high = np.zeros_like(adc, dtype=np.float32)
    
    adc_low[0] = -1e2
    adc_high[-1] = 1e2
    
    for s in range(len(adc) - 1):
        adc_high[s] = (adc[s] + adc[s + 1]) / 2
        adc_low[s + 1] = (adc[s] + adc[s + 1]) / 2

    return adc_low, adc_high
    
#########################

def adc_floor(adc):
    # make sure you pick the right type, zeros_like copies type.
    adc_thresh = np.zeros_like(adc, dtype=np.float32)
    
    for s in range(len(adc) - 1):
        adc_thresh[s] = (adc[s] + adc[s + 1]) / 2

    adc_thresh[-1] = adc[-1]
    
    return adc_thresh

#########################

def exp_err(s, p, var, adc, rpr, row):
    assert (np.all(p <= 1.))
    assert (len(s) == len(p))

    adc = sorted(adc)
    adc = np.reshape(adc, (-1, 1))
    adc_low, adc_high = adc_range(adc)

    pe = norm.cdf(adc_high, s, var * np.sqrt(s) + 1e-6) - norm.cdf(adc_low, s, var * np.sqrt(s) + 1e-6)
    e = s - adc
    
    # print (s.flatten())
    # print (adc.flatten())
    # print (e)
    # print (np.round(p * pe * e, 2))
    # print (adc_low.flatten())
    # print (adc_high.flatten())

    mu = np.sum(p * pe * e)
    std = np.sqrt(np.sum(p * pe * (e - mu) ** 2))

    mu = mu * row
    std = np.sqrt(std ** 2 * row)

    # print (rpr, (np.sum(np.absolute(e)), np.sum(pe), np.sum(p)), (mu, std))
    print (rpr, (mu, std), adc.flatten())
    
    return mu, std

#########################

class Conv(Layer):
    def __init__(self, input_size, filter_size, pool, stride, pad1, pad2, params, weights, relu_flag=True):
        self.layer_id = Layer.layer_id
        Layer.layer_id += 1
        self.weight_id = Layer.weight_id
        Layer.weight_id += 1

        self.input_size = input_size
        self.xh, self.xw, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.p = pool
        self.s = stride
        self.p1 = pad1
        self.p2 = pad2
        self.relu_flag = relu_flag

        self.params = params.copy()

        assert (self.s == 1 or self.p == 1)

        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        self.yw = self.yh

        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['y']
        # check shape
        assert(np.shape(self.w) == self.filter_size)
        assert(np.shape(self.b) == (self.fn,))
        assert(np.shape(self.q) == ())
        # cast as int
        self.w = self.w.astype(int)
        self.b = self.b.astype(int)
        self.q = self.q.astype(int)
        # q must be larger than 0
        # assert(self.q > 0)

        maxval = pow(2, self.params['bpw'] - 1)
        minval = -1 * maxval

        assert (np.all(self.w >= minval))
        assert (np.all(self.w <= maxval))

        self.wb = self.transform_weights()
        nwl, _, nbl, _ = np.shape(self.wb) 
        self.factor = nwl * nbl
        self.nwl = nwl
        self.nbl = nbl

        #########################

    def init(self, params):
        self.params.update(params)
        
        if self.params['rpr_alloc'] == 'centroids':
            self.params['var'] = lut_var(params['sigma'], 64)
            self.params['rpr'] = self.profile_rpr()

        elif self.params['rpr_alloc'] == 'dynamic':
            # self.params['var'] = lut_var_dyn(params['sigma'], 64)
            self.params['var'] = lut_var(params['sigma'], 64)
        
            ## TODO: cant this be "self.wb" and cant we throw it in a different function ??
            w_offset = self.w + self.params['offset']
            wb = []
            for bit in range(self.params['bpw']):
                wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
            wb = np.stack(wb, axis=-1)

            wb_cols = np.reshape(wb, (self.fh * self.fw * self.fc, self.fn, self.params['bpw']))
            col_density = np.mean(wb_cols, axis=0)

            nrow = self.fh * self.fw * self.fc
            p = np.max(col_density, axis=0)
            self.params['rpr'] = dynamic_rpr(nrow=nrow, p=p, q=self.q, params=self.params)
            # print (self.params['rpr'])

        else:
            assert (False)

    def set_profile_adc(self, counts):
        self.all_counts = counts[self.layer_id]

    def profile_adc(self, x):
        rpr_low = 1
        rpr_high = 64
        patches = self.transform_inputs(x)
        _, self.all_counts = profile(patches, self.wb, (self.yh * self.yw, self.fn), rpr_low, rpr_high, self.params)
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y_ref = self.act(y_ref)
        return y_ref, {self.layer_id: self.all_counts}

    def set_block_alloc(self, alloc):        
        nblock = np.sum(alloc)
        block_map = np.zeros(shape=nblock)
        block = 0
        for wl in range(self.nwl):
            for d in range(alloc[wl]):
                block_map[block] = wl
                block += 1

        self.block_map = np.ascontiguousarray(block_map.flatten(), np.int32)
        self.block_alloc = alloc
        self.nblock = np.sum(alloc)

    def set_layer_alloc(self, layer_alloc):
        pass
        
    def weights(self):
        return [self]

    def act(self, y):
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.round(y)
        y = np.clip(y, -128, 127)
        return y

    def forward(self, x, profile=False):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        y = self.act(y)
        y_ref = self.act(y_ref)

        y_min = np.min(y_ref)
        y_max = np.max(y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        # assert (self.s == 1)
        
        print ('y_mean', y_mean, 'y_std', y_std, 'y_max', y_max, 'y_min', y_min)
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        # results = {}
        results['id']    = self.weight_id
        results['nmac']  = self.nmac
        results['std']   = y_std
        results['mean']  = y_mean
        
        nwl, _, nbl, _ = np.shape(self.wb)
        
        if self.params['alloc'] == 'block': 
            # the sum here is confusing, since for layer 1, block performs better with less arrays.
            # but it actually makes sense.
            results['array'] = np.sum(self.block_alloc) * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall']))
                    
        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            print ('%d: alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_id, self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall']))

        ########################

        y = y_ref
        # assert (y_std <= 0)
        
        '''
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.round(y)
        y = np.clip(y, -128, 127)
        '''
        ########################

        return y, [results]
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        patches = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(patches)
        
        #########################
        
        if   self.params['alloc'] == 'block': alloc = self.block_map
        elif self.params['alloc'] == 'layer': assert (False)
        
        if self.params['rpr_alloc'] == 'centroids':
            y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], self.nblock, alloc, self.adc_state, self.adc_thresh, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
            y = y / 4
        elif self.params['rpr_alloc'] == 'dynamic':
            y, metrics = pim_dyn(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], self.nblock, alloc, self.params)
            y = np.reshape(y, (yh, yw, self.fn))
            # may have (*4) problems somewhere.
        else:
            assert (False)
        
        # we shud move this into forward, do it after the y - y_ref. 
        assert(np.all(np.absolute(y) < 2 ** 23))

        #########################
        
        # metrics = adc {1,2,3,4,5,6,7,8}, cycle, ron, roff, wl
        results = {}
        results['adc']   = metrics[0:8]
        results['cycle'] = metrics[8]
        results['ron']   = metrics[9]
        results['roff']  = metrics[10]
        results['wl']    = metrics[11]
        results['stall'] = metrics[12]
        results['block_cycle'] = metrics[13:]
        results['density'] = np.count_nonzero(patches) / np.prod(np.shape(patches)) * (self.params['wl'] / min(self.fh * self.fw * self.fc, self.params['wl']))
        results['block_density'] = np.count_nonzero(patches, axis=(0,2,3)) / (npatch * self.params['wl'] * self.params['bpa'])
        
        #########################
        
        return y, results
        
    def transform_inputs(self, x):
    
        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        #########################
        
        x = np.pad(array=x, pad_width=[[self.p1,self.p2], [self.p1,self.p2], [0,0]], mode='constant')
        patches = []
        for h in range(yh):
            for w in range(yw):
                patch = x[h*self.s:(h*self.s+self.fh), w*self.s:(w*self.s+self.fw), :]
                patch = np.reshape(patch, self.fh * self.fw * self.fc)
                patches.append(patch)
                
        #########################
        
        patches = np.stack(patches, axis=0)
        pb = []
        for xb in range(self.params['bpa']):
            pb.append(np.bitwise_and(np.right_shift(patches.astype(int), xb), 1))
        
        patches = np.stack(pb, axis=-1)
        npatch, nrow, nbit = np.shape(patches)
        
        #########################
        
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(npatch, self.params['wl'] - (nrow % self.params['wl']), self.params['bpa']))
            patches = np.concatenate((patches, zeros), axis=1)
            
        patches = np.reshape(patches, (npatch, -1, self.params['wl'], self.params['bpa']))
        
        #########################
        
        return patches
        
    def transform_weights(self):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        w_offset = np.copy(self.w) + self.params['offset']
        w_matrix = np.reshape(w_offset, (self.fh * self.fw * self.fc, self.fn))
        wb = []
        for bit in range(self.params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_matrix, bit), 1))
        wb = np.stack(wb, axis=-1)
        
        ########################
        
        nrow, ncol, nbit = np.shape(wb)
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(self.params['wl'] - (nrow % self.params['wl']), ncol, nbit))
            wb = np.concatenate((wb, zeros), axis=0)

        nrow, ncol, nbit = np.shape(wb)
        wb = np.reshape(wb, (-1, self.params['wl'], ncol, nbit))
        
        ########################

        nwl, wl, ncol, nbit = np.shape(wb)
        # wb = np.transpose(wb, (0, 1, 3, 2))
        wb = np.reshape(wb, (nwl, self.params['wl'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['wl'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['wl'], -1, self.params['bl']))

        ########################

        return wb
                
    def profile_rpr(self):

        rpr_low = 1
        rpr_high = 64
            
        self.adc_state = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
        self.adc_thresh = np.zeros(shape=(rpr_high + 1, self.params['adc'] + 1))
        
        rpr_dist = {}
        for rpr in range(rpr_low, rpr_high + 1):
            # values, counts, centroids = self.dist(x=x, rpr=rpr)
            counts = self.all_counts[rpr][0:rpr+1]
            values = np.array(range(rpr+1))
            
            if rpr <= self.params['adc']:
                centroids = np.arange(0, self.params['adc'] + 1, step=1, dtype=np.float32)
            else:
                centroids = kmeans(values=values, counts=counts, n_clusters=self.params['adc'] + 1)
                centroids = sorted(centroids)
            
            # p = counts / np.cumsum(counts)
            p = counts / np.sum(counts)
            s = values

            nrow = self.fh * self.fw * self.fc
            p_avg = 1. # TODO: we need to set this back to actual p_avg

            mu, std = exp_err(s=s, p=p, var=self.params['sigma'], adc=centroids, rpr=rpr, row=np.ceil(p_avg * nrow / rpr))
            rpr_dist[rpr] = {'mu': mu, 'std': std, 'centroids': centroids}
            
            self.adc_state[rpr] = 4 * np.array(centroids)
            self.adc_thresh[rpr] = adc_floor(centroids)
            
            if rpr == 1:
                print (self.adc_thresh[rpr])
                self.adc_thresh[rpr][0] = 0.2
            
        # def rpr(nrow, p, q, params):
        rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                rpr_lut[xb][wb] = self.params['adc']
            
        if not (self.params['skip'] and self.params['cards']):
            '''
            for key in sorted(rpr_lut.keys()):
                print (key, rpr_lut[key])
            print (np.average(list(rpr_lut.values())))
            '''
            return rpr_lut
        
        # counting cards:
        # ===============
        # TODO: we need to account for post processing.
        # (y - y_ref) also has (relu, bias) that we are ignoring.
        for wb in range(self.params['bpw']):
            for xb in range(self.params['bpa']):
                for rpr in range(rpr_low, rpr_high + 1):
                
                    scale = 2**wb * 2**xb
                    mu, std = rpr_dist[rpr]['mu'], rpr_dist[rpr]['std']
                    
                    # e = (scale / self.q) * 64 * std
                    # e_mu = (scale / self.q) * 64 * mu
                    e = (scale / self.q) * 5 * std
                    e_mu = (scale / self.q) * 5 * mu
                    # print (scale, e, e_mu)
                    
                    if rpr == rpr_low:
                        rpr_lut[xb][wb] = rpr
                    if (e < 1.) and (np.absolute(e_mu) < 1.):
                        rpr_lut[xb][wb] = rpr

        '''
        for key in sorted(rpr_lut.keys()):
            print (key, rpr_lut[key])
        print (np.average(list(rpr_lut.values())))
        '''
        return rpr_lut


        
        
        
