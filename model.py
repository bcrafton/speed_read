
import numpy as np
np.set_printoptions(threshold=1000)

class model:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        
        y = [None] * num_examples
        psum = [0] * num_examples
        for ii in range(num_examples):
            y[ii] = x[ii]
            for layer in self.layers:
                y[ii], p = layer.forward(x=y[ii])
                psum[ii] += p

        return y, psum

    


        
        
        
        
        
        
        
        
