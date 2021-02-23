
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

#########################

class Bert:
    def __init__(self, array_params):
        self.array_params = array_params
        np.random.seed(0)
        weights = np.load('../bert.npy', allow_pickle=True).item()
        self.embed = EmbedLayer(weights['embed'])
        self.encoder = []
        for l in range(12):
            self.encoder.append(BertLayer(weights['encoder'][l]))
        self.pooler = LinearLayer(weights['pool'])
        self.classifier = LinearLayer(weights['class'])

    def init(self, params):
        pass

    def set_profile_adc(self, counts):
        pass

    def profile_adc(self, x):
        pass

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
        h = h[:, 0]
        p = np.tanh(self.pooler.forward(h))
        o = self.classifier.forward(p)
        return o, o, {}

    def set_layer_alloc(self):
        pass

    def set_block_alloc(self):
        pass

#########################






        
        
        
