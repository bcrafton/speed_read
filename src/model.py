
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

    def set_profile_adc(self, counts):
        assert (counts['wl'] == self.array_params['wl'])
        assert (counts['max_rpr'] == self.array_params['max_rpr'])
        for layer in self.layers:
            layer.set_profile_adc(counts)

    def profile_adc(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)
        counts = {}

        args_list = {}
        for example in range(num_examples):
            y = x[example]
            for layer in range(num_layers):
                y, args, ratio, nrow = self.layers[layer].profile_adc(x=y)
                args_list.update(args)

                for l in ratio.keys():
                    if l not in counts.keys():
                        counts[l] = {'adc': 0., 'row': 0., 'sat': 0., 'ratio': 0.}
                    counts[l]['ratio'] += ratio[l] / num_examples

                for l in nrow.keys():
                    if l not in counts.keys():
                        counts[l] = {'adc': 0., 'row': 0., 'sat': 0., 'ratio': 0.}
                    counts[l]['row'] += nrow[l] / num_examples

        manager = multiprocessing.Manager()
        thread_results = []

        keys = list(args_list.keys())
        total = len(keys)
        nthread = 8
        for run in range(0, total, nthread):

            threads = []
            for parallel_run in range(min(nthread, total - run)):
                thread_result = manager.dict()
                thread_results.append(thread_result)

                id = keys[run + parallel_run]
                args = args_list[id]
                args = args + (id, thread_result)

                t = multiprocessing.Process(target=profile, args=args)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        for result in thread_results:
            for key in result.keys():
                counts[key]['adc'] += result[key]['adc']
                # counts[key]['row'] += result[key]['row']
                counts[key]['sat'] += result[key]['sat']

        counts['wl'] = self.array_params['wl']
        counts['max_rpr'] = self.array_params['max_rpr']
        return counts

    def forward(self, x, y):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)

        correct = 0
        results = []
        for example in range(num_examples):
            out, out_ref = x[example], x[example]
            for layer in range(num_layers):
                out, out_ref, result = self.layers[layer].forward(x=out, x_ref=out_ref)
                for r in result:
                    r['example'] = example
                    results.append(r)
            # correct += (np.argmax(out) == y[example])
        # print (correct / num_examples)
        return out, out_ref, results

#########################






        
        
        
