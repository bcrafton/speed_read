
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

    def profile(self, x):
        y = x[0] # example 1
        args = []
        for layer in self.layers:
            y, arg = layer.profile(y)
            args.extend(arg)

        nthread = 24
        for run in range(0, len(args), nthread):
            threads = []
            for parallel_run in range(min(nthread, len(args) - run)):
                t = multiprocessing.Process(target=profile, args=args[run + parallel_run])
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

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






        
        
        
