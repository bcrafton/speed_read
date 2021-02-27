
import math
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

from var import *
from conv_utils import *
from scipy.stats import norm, binom
from AA import array_allocation
from cprofile import profile

from bert_layers import *

import sys, os, psutil
from size import get_size

#########################

class Bert:
    def __init__(self, params):
        self.params = params
        np.random.seed(0)
        weights = np.load('../bert1.npy', allow_pickle=True).item()

        # embedding
        self.embed = EmbedLayer(weights['embed'])
        
        # encoder
        self.encoder = []
        for l in range(12):
            self.encoder.append(BertLayer(params=params, weights=weights['encoder'][l]))
        
        # pooler
        self.pooler = LinearLayer(params=params, weights=weights['pool'])
        self.classifier = LinearLayer(params=params, weights=weights['class'])

    def init(self, params):
        self.table = {}
        self.params.update(params)
        self.embed.init(params, self.table)
        for layer in self.encoder:
            layer.init(params, self.table)
        self.pooler.init(params, self.table)
        self.classifier.init(params, self.table)
        #############################################
        N = len(self.table.keys())
        total_mac = np.zeros(shape=N)
        total_array = np.zeros(shape=N)
        perf = np.zeros(shape=N)
        for id in self.table.keys():
            total_mac[id] = self.table[id].get('total_mac')
            total_array[id] = self.table[id].get('total_array')
            perf[id] = self.params['bl'] // self.params['bpw']
        #############################################
        duplicate = array_allocation(self.params['narray'], total_mac, total_array, perf, self.params)
        duplicate = duplicate / total_array
        duplicate = duplicate.astype(np.int32)
        for id in self.table.keys():
            self.table[id].set('duplicate', duplicate[id])
        #############################################
        

    # make 1 set function
    def set_profile_adc(self, counts):
        assert (counts['wl'] == self.params['wl'])
        assert (counts['max_rpr'] == self.params['max_rpr'])
        for l in range(12):
            h = self.encoder[l].set_profile_adc(counts)
        self.pooler.set_profile_adc(counts)
        self.classifier.set_profile_adc(counts)

    def profile_adc(self, x):
        counters = {'adc': {}, 'row': {}, 'ratio': {}, 'sat': {}}
        (ids, tok, mask) = x
        embed = self.embed.forward(ids, tok)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        h = embed
        for l in range(12):
            h = self.encoder[l].profile_adc(h, mask, counters)
        batch, word, vec = np.shape(h)
        h = h[:, 0, :].reshape(batch, 1, vec)
        p = np.tanh(self.pooler.profile_adc(h, counters))
        o = self.classifier.profile_adc(p, counters)

        manager = multiprocessing.Manager()
        thread_results = []

        keys = list(counters['adc'].keys())
        total = len(keys)
        nthread = 32

        for run in range(0, total, nthread):

            threads = []
            for parallel_run in range(min(nthread, total - run)):
                thread_result = manager.dict()
                thread_results.append(thread_result)

                id = keys[run + parallel_run]
                args = counters['adc'][id]
                args = args + (id, thread_result)

                t = multiprocessing.Process(target=profile, args=args)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
        
        for result in thread_results:
            for key in result.keys():
                counters['adc'][key] = result[key]['adc']
                counters['sat'][key] = result[key]['sat']

        counters['wl'] = self.params['wl']
        counters['max_rpr'] = self.params['max_rpr']
        return counters

    def profile(self, x):
        results = {}
        ##########################################
        (ids, tok, mask) = x
        embed = self.embed.forward(ids, tok)
        ##########################################
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        ##########################################
        h = embed
        for l in range(12):
            h = self.encoder[l].forward((h, mask), results)
        ##########################################
        batch, word, vec = np.shape(h)
        h = h[:, 0, :].reshape(batch, 1, vec)
        p = np.tanh(self.pooler.forward(h, results))
        ##########################################
        o = self.classifier.forward(p, results)
        ##########################################
        ##########################################
        ##########################################
        N = len(results.keys())
        perf = np.zeros(shape=N)
        nmac = np.zeros(shape=N)
        factor = np.zeros(shape=N)
        for id in results.keys():
            nmac[id]   = results[id]['nmac']
            factor[id] = results[id]['nwl'] * results[id]['nbl']
            perf[id]   = nmac[id] / factor[id] / (results[id]['cycle'] * results[id]['duplicate'])
        ##########################################
        self.allocate(nmac, factor, perf)
        ##########################################
        
    def forward(self, x, y):
        results = {}
        ##########################################
        (ids, tok, mask) = x
        embed = self.embed.forward(ids, tok)
        ##########################################
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        ##########################################
        h = embed
        for l in range(12):
            h = self.encoder[l].forward((h, mask), results)
        ##########################################
        batch, word, vec = np.shape(h)
        h = h[:, 0, :].reshape(batch, 1, vec)
        p = np.tanh(self.pooler.forward(h, results))
        ##########################################
        o = self.classifier.forward(p, results)
        ##########################################
        return o, results

    def allocate(self, nmac, factor, perf=None):
        if perf == None: perf = np.ones_like(nmac) * (self.params['bl'] // self.params['bpw'])
        alloc = array_allocation(self.params['narray'], nmac, factor, perf, self.params)
        assert (np.sum(alloc) <= self.params['narray'])
        print (alloc)
            

#########################






        
        
        
