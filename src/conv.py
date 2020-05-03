
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from conv_utils import *
from cdot import *
from dot_ref import *
from defines import *
from var import *

from layers import *

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

        assert (self.s == 1 or self.p == 1)
        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        maxval = pow(2, params['bpw'] - 1)
        minval = -1 * maxval

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['y']
        assert (np.all(self.w >= minval))
        assert (np.all(self.w <= maxval))
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
        
        #########################
        
        self.wb = self.transform_weights()
        nwl, _, nbl, _ = np.shape(self.wb) 
        self.factor = nwl * nbl
        
        self.nwl = nwl
        self.nbl = nbl
        
        #########################

        self.params = params.copy()
        self.params['var'] = lut_var(params['sigma'], 32)
        
        #########################
        '''
        w_offset = self.w + params['offset']
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_offset, bit), 1))
        wb = np.stack(wb, axis=-1)

        wb_cols = np.reshape(wb, (self.fh * self.fw * self.fc, self.fn, params['bpw']))
        col_density = np.mean(wb_cols, axis=0)

        nrow = self.fh * self.fw * self.fc
        p = np.max(col_density, axis=0)
        # [0.74829932 0.74829932 0.74829932 0.74829932 0.74829932 0.74829932 0.74829932 0.79591837]
        # this should be specific to the block as well.
        
        #########################
        
        self.params['rpr'] = rpr(nrow=nrow, p=p, q=self.q, params=self.params)
        '''
        #########################

    def set_block_alloc(self, block_alloc):
        self.block_alloc = block_alloc

    def set_layer_alloc(self, layer_alloc):
        self.layer_alloc = layer_alloc
        
    def weights(self):
        return [self]

    def forward(self, x):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        y_min = np.min(y - y_ref)
        y_max = np.max(y - y_ref)
        y_mean = np.mean(y - y_ref)
        y_std = np.std(y - y_ref)
        # assert (self.s == 1)
        
        # print ('y_mean', y_mean, 'y_std', y_std)
        
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
            print ('alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (np.sum(self.block_alloc), nbl, nbl * np.sum(self.block_alloc), results['nmac'], results['cycle'], results['stall']))
                    
        elif self.params['alloc'] == 'layer': 
            results['array'] = self.layer_alloc * nwl * nbl
            print ('alloc: %d*%d=%d nmac %d cycle: %d stall: %d' % (self.layer_alloc, nwl * nbl, nwl * nbl * self.layer_alloc, results['nmac'], results['cycle'], results['stall']))

        ########################

        # y = y_ref
        # assert (y_std <= 0)

        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        y = avg_pool(y, self.p)
        y = y / self.q
        y = np.round(y)
        y = np.clip(y, -128, 127)
        
        ########################

        return y, [results]
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        patches = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(patches)
        
        #########################
        
        if   self.params['alloc'] == 'block': alloc = self.block_alloc
        elif self.params['alloc'] == 'layer': alloc = self.layer_alloc
        
        y, metrics = pim(patches, self.wb, (yh * yw, self.fn), self.params['var'], self.params['rpr'], alloc, self.params)
        y = np.reshape(y, (yh, yw, self.fn))
        
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
                patch = np.reshape(x[h*self.s:(h*self.s+self.fh), w*self.s:(w*self.s+self.fw), :], -1)
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
                
    def dist(self, x):
        x = self.transform_inputs(x)
        
        npatch, nwl, wl, bpa = np.shape(x)
        nwl, wl, nbl, bl = np.shape(self.wb)
        
        x = np.transpose(x, (0,3,1,2))
        x = np.reshape(x, (npatch * bpa, nwl, wl))

        print (np.shape(x), np.std(x))
        print (np.shape(self.wb), np.std(self.wb))

        #########################
        
        psums = [[] for _ in range(nwl)] 

        for p in range(npatch):
            for i in range(nwl):
                wlsum = 0
                psum = np.zeros(shape=(nbl, bl))
                
                for j in range(wl):
                
                    if x[p][i][j]:
                        wlsum += 1
                        psum += self.wb[i][j]
                        
                    # damn it. rpr is 2d array ...
                    # well we cannot control xb
                    # but we can do what we did before with p
                    # and split them up correctly by wb.
                    
                    # well we actually can control xb
                    # because we are feeding in the activations we have
                    # so - i wud say we want to evaluate all the rpr's and collect 
                    # distributions outright.
                    if wlsum == 12: # self.params['rpr']:
                        wlsum = 0
                        psums[i].append(psum)
                        psum = np.zeros(shape=(nbl, bl))
                
                psums[i].append(psum)
        
        #########################

        x = psums[0]
        x = np.array(x)
        x = np.reshape(x[:, :, :], (-1, 1))

        values, counts = np.unique(x, return_counts=True)
        # plt.hist(x)
        # plt.show()
        print (values)
        print (counts)

        #########################

        kmeans = KMeans(n_clusters=self.params['adc'], init='k-means++', max_iter=300, n_init=5, random_state=0)
        kmeans.fit(x)

        centroids = np.round(kmeans.cluster_centers_[:, 0], 2)
        print (centroids)
        
        assert (False)
                
        #########################

    def profile_rpr(self, x):

    # def rpr(nrow, p, q, params):
        rpr_lut = np.zeros(shape=(8, 8), dtype=np.int32)
        for wb in range(params['bpw']):
            for xb in range(params['bpa']):
                rpr_lut[xb][wb] = params['adc']
            
        if not (params['skip'] and params['cards']):
            '''
            for key in sorted(rpr_lut.keys()):
                print (key, rpr_lut[key])
            print (np.average(list(rpr_lut.values())))
            '''
            return rpr_lut
        
        # counting cards:
        # ===============
        for wb in range(params['bpw']):
            for xb in range(params['bpa']):
                rpr_low = 1
                rpr_high = 16
                for rpr in range(rpr_low, rpr_high + 1):
                    scale = 2**(wb - 1) * 2**(xb - 1)
                    mu, std = prob_err(p[wb], params['sigma'], params['adc'], rpr, np.ceil(nrow / rpr))
                    e = (scale / q) * 5 * std
                    e_mu = (scale / q) * mu

                    if rpr == rpr_low:
                        rpr_lut[xb][wb] = rpr
                    if (e < 1.) and (np.absolute(e_mu) < 0.15):
                    # if e < 1.:
                        rpr_lut[xb][wb] = rpr

        '''
        for key in sorted(rpr_lut.keys()):
            print (key, rpr_lut[key])
        print (np.average(list(rpr_lut.values())))
        '''
        return rpr_lut



        
        
        