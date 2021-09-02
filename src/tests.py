
import copy
import numpy as np

#######################################################

def perms(param):
    params = [param]
    
    for key in param.keys():
        val = param[key]
        if type(val) == list:
            new_params = []
            for ii in range(len(val)):
                for jj in range(len(params)):
                    new_param = copy.copy(params[jj])
                    new_param[key] = val[ii]
                    new_params.append(new_param)
                    
            params = new_params
            
    return params
    
#######################################################

def CC():

    rpr64 = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64])
    adc64 = np.array([1])
    sar64 = np.array([0, 2, 3, 4, 5, 6])
    Ns    = np.array([1])

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc_mux': 8,
    'wl': 256,
    'bl': 256,
    'offset': 128,
    'max_rpr': 64
    }

    arch_params1 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 9],
    'cards': [1],
    'profile': [0],
    'thresh': [0.10],
    'method': ['kmeans'],
    'adc': 64,
    'lrs': [0.01],
    'hrs': [0.02],
    'area': [1e9],
    'rprs': [rpr64],
    'adcs': [adc64],
    'sars': [sar64],
    'Ns':   [Ns]
    }

    arch_params1 = perms(arch_params1)
    arch_params = arch_params1
    return array_params, arch_params
    
#######################################################
    
def BB():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 128,
    'bl': 128,
    'offset': 128,
    'max_rpr': 64,
    }
    
    arch_params = {
    'skip': [0, 1],
    'alloc': ['layer', 'block'],
    'narray': [5472, 2 ** 13, 1.5 * 2 ** 13, 2 ** 14, 1.5 * 2 ** 14],
    'sigma': [0.01],
    'cards': [0],
    'profile': [0, 1],
    'rpr_alloc': ['dynamic'],
    'thresh': [1.00]
    }

    arch_params = perms(arch_params)
    
    return array_params, arch_params
    
#######################################################
    
def Thresh():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 128,
    'bl': 128,
    'offset': 128,
    'max_rpr': 64,
    }
    
    arch_params1 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 13],
    'cards': [1],
    'profile': [1],
    'rpr_alloc': ['static'],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'thresh': [0.5, 2.00, 5.00]
    }

    arch_params2 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 13],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'cards': [0],
    'profile': [1],
    'rpr_alloc': ['dynamic'],
    'thresh': [1.00]
    }

    arch_params1 = perms(arch_params1)
    arch_params2 = perms(arch_params2)
    arch_params = arch_params1 + arch_params2
    return array_params, arch_params
    
#######################################################
    
def CE():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 128,
    'bl': 128,
    'offset': 128,
    'max_rpr': 64,
    }
    
    arch_params1 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12, 1.5 * 2 ** 12, 2 ** 13, 1.5 * 2 ** 13],
    'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
    'cards': [1],
    'profile': [1],
    'rpr_alloc': ['static'],
    'thresh': [1.00]
    }

    arch_params2 = {
    'skip': [1],
    'alloc': ['layer'],
    'narray': [2 ** 12, 1.5 * 2 ** 12, 2 ** 13, 1.5 * 2 ** 13],
    'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
    'cards': [1],
    'profile': [0],
    'rpr_alloc': ['static'],
    'thresh': [1.00]
    }

    arch_params1 = perms(arch_params1)
    arch_params2 = perms(arch_params2)
    arch_params = arch_params1 + arch_params2
    return array_params, arch_params

#######################################################

def Simple():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 256,
    'bl': 256,
    'offset': 128,
    'max_rpr': 16,
    'sar': True,
    'max_step': 2
    }

    arch_params = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [128],
    'lrs': [0.07], 
    'hrs': [0.03],
    'cards': [0, 1],
    'profile': [0],
    'rpr_alloc': ['static'],
    'thresh': [0.1],
    }

    arch_params = perms(arch_params)
    return array_params, arch_params

#######################################################

def dac2():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 256,
    'bl': 256,
    'offset': 128,
    'max_rpr': 64,
    }

    arch_params = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12],
    'cards': [1],
    'profile': [1],
    'rpr_alloc': ['static'],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'thresh': [0.25, 1.00, 2.00]
    }

    arch_params = perms(arch_params)
    return array_params, arch_params

#######################################################






