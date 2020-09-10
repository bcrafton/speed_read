
import math
import numpy as np
import matplotlib.pyplot as plt

#########################

# PIMConfig
# ArrayConfig
# Config -> KmeansConfig, DynamicConfig
# what is a good name for this ? 

class Config:
    
    def __init__(self, low, high, params, profile, nrow, q):
        self.low = low
        self.high = high
        self.params = params
        self.profile = profile
        self.nrow = nrow
        self.q = q

    '''
    def dynamic(self):
        pass
        
    def kmeans(self):
        pass
        
    def static(self):
        pass
    '''
    
    def rpr(self):
        pass
        
#########################







        
        
        
