
import numpy as np
np.set_printoptions(threshold=1000)

class model:
    def __init__(self, layers : tuple):
        self.num_layers = len(layers)
        self.layers = layers

    def forward(self, X):
        A = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])

        return A[self.num_layers-1]

    


        
        
        
        
        
        
        
        
