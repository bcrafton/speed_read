
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
    def __init__(self, array_params):
        self.array_params = array_params
        np.random.seed(0)
        weights = np.load('../bert8b.npy', allow_pickle=True).item()

        # embedding
        self.embed = EmbedLayer(weights['embed'])
        
        # encoder
        self.encoder = []
        for l in range(12):
            self.encoder.append(BertLayer(params=array_params, weights=weights['encoder'][l]))
        
        # pooler
        self.pooler = LinearLayer(params=array_params, weights=weights['pool'])
        self.classifier = LinearLayer(params=array_params, weights=weights['class'])

    def init(self, params):
        self.embed.init(params)
        for layer in self.encoder:
            layer.init(params)
        self.pooler.init(params)
        self.classifier.init(params)

    def set_profile_adc(self, counts):
        pass

    def profile_adc(self, x):
        counters = {'adc': {}, 'row': {}, 'ratio': {}}
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
        nthread = 10

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

        counts = {}
        for result in thread_results:
            for key in result.keys():
                counts[key]['adc'] += result[key]['adc']

        for key in counters['row'].keys():
            counts[key]['row'] = counters['row']
        
        for key in counters['ratio'].keys():
            counts[key]['ratio'] = counters['ratio']

        counts['wl'] = self.array_params['wl']
        counts['max_rpr'] = self.array_params['max_rpr']
        return counts

    def profile(self, x):
        pass

    def forward(self, x, y):
        (ids, tok, mask) = x
        embed = self.embed.forward(ids, tok)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        h = embed
        for l in range(12):
            h = self.encoder[l].forward(h, mask)
        batch, word, vec = np.shape(h)
        h = h[:, 0, :].reshape(batch, 1, vec)
        p = np.tanh(self.pooler.forward(h))
        o = self.classifier.forward(p)
        return o, o, {}

    def set_layer_alloc(self):
        pass

    def set_block_alloc(self):
        pass

#########################






        
        
        
