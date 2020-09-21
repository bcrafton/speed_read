
import math
import numpy as np
import matplotlib.pyplot as plt

from var import *
from conv_utils import *

from scipy.stats import norm, binom

from AA import array_allocation

#########################

class Model:
    def __init__(self, layers, array_params):
        self.layers = layers
        self.nlayer = len(self.layers)
        self.array_params = array_params
        np.random.seed(0)

    def init(self, params):
        self.params = params
        for layer in self.layers:
            layer.init(params)

        self.weights = []
        for layer in self.layers:
            self.weights.extend(layer.weights())
        self.nweight = len(self.weights)

        self.block_map = []
        self.nblock = 0
        for w, weight in enumerate(self.weights):
            self.block_map.append(slice(self.nblock, self.nblock + weight.nwl))
            self.nblock += weight.nwl

        self.mac_per_array_layer = [2.] * self.nweight
        self.set_layer_alloc()
        
        print ('nblock', self.nblock)
        self.mac_per_array_block = [2.] * self.nblock
        self.set_block_alloc()

    def set_profile_adc(self, counts):
        assert (counts['wl'] == self.array_params['wl'])
        for layer in self.layers:
            layer.set_profile_adc(counts)

    def profile_adc(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)
        counts = {}

        for example in range(num_examples):
            y = x[example]
            for layer in range(num_layers):
                y, adc_counts = self.layers[layer].profile_adc(x=y)
                assert (np.all((y % 1) == 0))
                counts.update(adc_counts)

        counts['wl'] = self.array_params['wl']
        return counts

    def profile(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        results = {}

        mac_per_array_layer = np.zeros(shape=(num_examples, self.nweight))
        mac_per_array_block = np.zeros(shape=(num_examples, self.nblock))
        
        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example], profile=True)
                assert (np.all((pred[example] % 1) == 0))
                for r in result:
                    mac_per_array_layer[example][r['id']] = (r['nmac'] / self.weights[r['id']].factor) / (r['cycle'] * self.weights[r['id']].layer_alloc)
                    mac_per_array_block[example][self.block_map[r['id']]] = (r['nmac'] / self.weights[r['id']].factor) / (r['block_cycle'])
                    
        self.mac_per_array_layer = np.mean(mac_per_array_layer, axis=0)
        self.mac_per_array_block = np.mean(mac_per_array_block, axis=0)
        
        if self.params['alloc'] == 'layer': 
            self.set_layer_alloc() # block alloc was failing when layer was selected, this is a bandaid.
        else:
            self.set_block_alloc()

    def forward(self, x, y):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        pred = [None] * num_examples
        # results = [[] for weight in range(self.nweight)] 
        
        '''
        results = {}
        for weight in range(self.nweight):
            results[weight] = []
        '''
        
        results = []
        
        for example in range(num_examples):
            pred[example] = x[example]
            for layer in range(num_layers):
                pred[example], result = self.layers[layer].forward(x=pred[example])
                assert (np.all((pred[example] % 1) == 0))
                '''
                for r in result:
                    results[r['id']].append(r)
                '''
                for r in result:
                    r['example'] = example
                    results.append(r)
                
        # this is dumb, just stick this in the results.
        # results['block_mac'] = self.mac_per_array_block
        # results['layer_mac'] = self.mac_per_array_layer

        return pred, results

    def set_layer_alloc(self):
        nmac = np.zeros(shape=self.nweight, dtype=np.int32)
        factor = np.zeros(shape=self.nweight, dtype=np.int32)
        for weight in range(self.nweight):
            nmac[weight] = self.weights[weight].nmac
            factor[weight] = self.weights[weight].factor
                
        # alloc = branch_and_bound(self.params['narray'], nmac, factor, self.mac_per_array_layer, self.params)
        alloc = array_allocation(self.params['narray'], nmac, factor, self.mac_per_array_layer, self.params)
        assert (np.sum(alloc) <= self.params['narray'])
        # assert (np.sum(alloc) == 2 ** 14)
        print ("%d / %d" % (np.sum(alloc), self.params['narray']))

        for weight in range(len(self.weights)):
            layer_alloc = alloc[weight] // self.weights[weight].factor
            self.weights[weight].set_layer_alloc(layer_alloc)

    def set_block_alloc(self):
        nmac = np.zeros(shape=self.nblock, dtype=np.int32)
        factor = np.zeros(shape=self.nblock, dtype=np.int32)
        block = 0
        for weight in range(self.nweight):
            nwl, _, nbl, _ = np.shape(self.weights[weight].wb) 
            for wl in range(nwl):
                nmac[block] = self.weights[weight].nmac // nwl
                factor[block] = nbl
                block += 1
                
        # alloc = branch_and_bound(self.params['narray'], nmac, factor, self.mac_per_array_block, self.params)
        alloc = array_allocation(self.params['narray'], nmac, factor, self.mac_per_array_block, self.params)
        assert (np.sum(alloc) <= self.params['narray'])

        block1 = 0
        for weight in range(self.nweight):
            block2 = block1 + self.weights[weight].nwl
            block_alloc = np.array(alloc[block1:block2]) // self.weights[weight].nbl
            self.weights[weight].set_block_alloc(block_alloc)
            block1 = block2

#########################






        
        
        
