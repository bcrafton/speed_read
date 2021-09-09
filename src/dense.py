
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
from static_rpr import static_rpr

#########################

class Dense(Layer):
    def __init__(self, size, params, weights, relu_flag=False):
        assert (False)

    def init(self, params):
        assert (False)

    def forward(self, x, x_ref):
        assert (False)

    def transform_inputs(self, x):

        x = np.reshape(x, self.input_size)

        #########################

        xb = []
        for bit in range(self.params['bpa']):
            xb.append(np.bitwise_and(np.right_shift(x.astype(int), bit), 1))
        
        xb = np.stack(xb, axis=-1)
        nrow, nbit = np.shape(xb)
        
        #########################
        
        if (nrow % self.params['wl']):
            zeros = np.zeros(shape=(npatch, self.params['wl'] - (nrow % self.params['wl']), self.params['bpa']))
            xb = np.concatenate((xb, zeros), axis=1)
            
        xb = np.reshape(xb, (1, -1, self.params['wl'], self.params['bpa']))
        
        #########################
        
        return xb
        
    def transform_weights(self):
        w_offset = np.copy(self.w) + self.params['offset']
        w_matrix = np.reshape(w_offset, (self.input_size, self.output_size))
        
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

        ########################

        return wb


        
        
        
