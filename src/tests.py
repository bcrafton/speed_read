
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

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 8,
    'adc_mux': 8,
    'wl': 256,
    'bl': 256,
    'offset': 128,
    'max_rpr': 16,
    }
    
    arch_params1 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'cards': [1],
    'profile': [1],
    'rpr_alloc': ['static'],
    'thresh': [0.1, 0.25, 0.5],
    }

    arch_params2 = {
    'alloc': ['block'],
    'narray': [2 ** 12],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'cards': [0],
    'profile': [1],
    'rpr_alloc': ['dynamic'],
    'thresh': [0.1],
    'skip': [0],
    }

    arch_params3 = {
    'alloc': ['block'],
    'narray': [2 ** 12],
    'sigma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
    'cards': [0],
    'profile': [1],
    'rpr_alloc': ['dynamic'],
    'thresh': [0.1],
    'skip': [1],
    }

    arch_params1 = perms(arch_params1)
    arch_params2 = perms(arch_params2)
    arch_params3 = perms(arch_params3)
    arch_params = arch_params1 + arch_params2 + arch_params3
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
    'narray': [2 ** 14],
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
    }

    arch_params = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12],
    'cards': [0],
    'profile': [0],
    'rpr_alloc': ['static'],
    'sigma': [0.05],
    'thresh': [1.00],
    'ABFT': [1]
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






