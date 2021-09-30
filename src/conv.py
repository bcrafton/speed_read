
import math
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans

from conv_utils import *
from dot_ref import *
from var import *

from layers import *
from cprofile import profile
from static_rpr import static_rpr
from cim import cim
from adc import confusion

#########################

class Conv(Layer):
    def __init__(self, input_size, filter_size, pool, stride, pad1, pad2, params, weights, relu_flag=True, quantize_flag=True):
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
        self.quantize_flag = quantize_flag

        self.params = params.copy()

        assert (self.s == 1 or self.p == 1)

        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        self.yw = self.yh

        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.xh * self.xw) // (self.s ** 2)

        self.w, self.b, self.q = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['q']
        # check shape
        assert(np.shape(self.w) == self.filter_size)
        assert(np.shape(self.b) == (self.fn,))
        assert(np.shape(self.q) == ())
        # cast as int
        assert(np.max(self.w) < 128 and np.min(self.w) >= -128)
        self.w = self.w.astype(np.int8)
        self.b = self.b.astype(np.float32)
        self.q = self.q.astype(np.float32)
        # q must be larger than 0
        if self.quantize_flag:
            assert(self.q > 0)
        self.params['q'] = self.q

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
        self.params['rpr'], self.params['comps'], self.params['sar'], self.params['N'], self.params['conf'], self.params['value'], self.params['exp_error'], self.params['exp_mean'], self.params['exp_p'] = static_rpr(self.layer_id, self.params, self.q)
        # print (self.params['rpr'])
        # print (self.params['comps'])
        # print (self.params['sar'])
        # print (self.params['N'])
        # print (self.params['conf'])
        # print (self.params['value'])


    def profile(self, x):
        # x
        patches = self.transform_inputs(x)
        # w 
        w = self.wb
        # y
        y_ref = conv_ref(x=x, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y_ref = self.act(y_ref, quantize_flag=self.quantize_flag)
        # y_shape
        y_shape = (self.yh * self.yw, self.c)

        return y_ref, [(self.layer_id, patches, self.wb, y_ref, y_shape, self.params)]

    def act(self, y, quantize_flag):
        y = y + self.b
        if self.relu_flag:
            y = relu(y)
        assert(self.p == 1)
        # y = avg_pool(y, self.p)
        if quantize_flag:
            y = y / self.q
            y = np.around(y)
            y = np.clip(y, -128, 127)
        return y

    def forward(self, x, x_ref):
        # 1) tensorflow to compute y_ref
        # 2) save {x,y1,y2,...} as tb from tensorflow 
        y_ref = conv_ref(x=x_ref, f=self.w, b=self.b, q=self.q, pool=self.p, stride=self.s, pad1=self.p1, pad2=self.p2, relu_flag=self.relu_flag)
        y, results = self.conv(x=x)

        mean = np.mean(y - y_ref)
        error = np.mean(np.absolute(y - y_ref))
        mse = np.sqrt(np.mean( (y - y_ref)**2 ))
        std = np.std(y - y_ref)
        results['cim_mean'] = mean
        results['cim_error'] = error
        results['cim_std'] = 0

        z = self.act(y, quantize_flag=True)
        z_ref = self.act(y_ref, quantize_flag=True)

        # nonzero = np.count_nonzero(z_ref) / np.prod(np.shape(z_ref))
        # print (nonzero)

        z_min = np.min(z_ref)
        z_max = np.max(z_ref)
        z_mean = np.mean(z - z_ref)
        z_std = np.std(z - z_ref)
        z_error = np.mean(np.absolute(z - z_ref))

        ref_error = error / self.q
        ref_mse = mse / self.q
        ref_mean = mean / self.q
        ref_std = std / self.q

        # plt.hist((y.flatten() - y_ref.flatten()) / self.q, bins=100)
        # plt.show()

        # print (self.error, self.mean)
        # print (error * self.ratio / self.q, mean * self.ratio / self.q)
        # print (error / self.q * self.ratio)
        # print (self.params['rpr'])

        results['id']       = self.weight_id
        results['layer_id'] = self.layer_id
        results['nmac']     = self.nmac
        results['std']      = ref_std
        results['mean']     = ref_mean
        results['error']    = ref_mse

        # count = (1024, 1, 8, 8, 2)
        # sar = (8, 8)
        # adc = (8, 8)
        cost = self.params['sar'].reshape(8, 8) * self.params['comps'].reshape(8, 8) + self.params['sar'].reshape(8, 8)
        results['energy'] = np.sum(cost * results['vmm_cycles'])

        nwl, _, nbl, _ = np.shape(self.wb)
        results['nwl'] = nwl
        results['nbl'] = nbl

        print ('lrs: %f cycle: %d energy: %d stall: %d mean: %0.4f mse: %0.4f std: %0.3f error: %0.3f' %
              (self.params['lrs'], results['cycle'], results['energy'], results['stall'], ref_mean, ref_mse, ref_std, ref_error))

        ########################

        y = self.act(y, quantize_flag=self.quantize_flag)
        y_ref = self.act(y_ref, quantize_flag=self.quantize_flag)

        ########################

        # accuracy: y_ref = y
        # perf:     y = y_ref
        # error:    None

        y = y_ref
        # y_ref = y

        ########################

        return y, y_ref, [results]
        
    def conv(self, x):

        yh = (self.xh - self.fh + self.s + self.p1 + self.p2) // self.s
        yw = yh
        
        patches = self.transform_inputs(x)
        npatch, nwl, wl, nbit = np.shape(patches)
        
        #########################

        y, metrics = cim(self.layer_id, patches, self.wb, self.params)
        y = np.reshape(y, (yh, yw, self.fn))

        #########################

        results = metrics
        results['rpr'] = self.params['rpr']
        results['sar'] = self.params['sar']
        results['N'] = self.params['N']
        results['area'] = self.params['area']
        results['comps'] = self.params['comps']

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

        w_matrix = np.reshape(np.copy(self.w), (self.fh * self.fw * self.fc, self.fn))
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
        wb = np.reshape(wb, (nwl, self.params['wl'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['wl'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['wl'], -1, self.params['bl']))
        wb = wb.astype(int)

        '''
        import matplotlib.pyplot as plt
        dist = np.sum(wb, axis=(0, 1)).flatten()
        plt.hist(dist)
        plt.show()
        # assert (False)
        '''
        '''
        dist = np.reshape(wb,   (1, 256, 2, 256))
        dist = np.reshape(dist, (1, 256, 2, 32, 8))
        pmf = np.mean(dist, axis=(0, 1, 2, 3))
        wbs = []
        for i in range(8):
            wb = np.random.choice(a=[0, 1], size=(1, 256, 2, 32), p=[1 - pmf[i], pmf[i]], replace=True)
            wbs.append(wb)
        wb = np.stack(wbs, axis=-1)

        import matplotlib.pyplot as plt
        dist = np.sum(wb, axis=(0, 1)).flatten()
        plt.hist(dist)
        plt.show()
        # assert (False)

        scale = np.array([1,2,4,8,16,32,64,-128])
        w = np.sum(scale * wb, axis=4)
        w = np.reshape(w, (nwl * wl, ncol // nbit))
        w = w[0:self.fh * self.fw * self.fc, 0:self.fn]
        w = np.reshape(w, (self.fh, self.fw, self.fc, self.fn))
        self.w = w

        wb = np.reshape(wb, (1, 256, 2, 256))
        '''

        return wb
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
