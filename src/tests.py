
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
    sar64 = np.array([1, 2, 3, 4, 5, 6])
    Ns    = np.array([1])

    adc64_area = np.array([1])
    sar64_area = np.array([0, 1, 1.25, 1.50, 1.75, 2.00])

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
    'method': ['soft'],
    'adc': 64,
    'lrs': [0.01],
    'hrs': [0.02],
    'area': [16],
    'rprs': [rpr64],
    'adcs': [adc64],
    'sars': [sar64],
    'Ns':   [Ns],
    'adc_area': [adc64_area],
    'sar_area': [sar64_area],
    'adc_energy': 1,
    'sar_energy': 1
    }

    arch_params1 = perms(arch_params1)
    arch_params = arch_params1
    return array_params, arch_params
    
#######################################################
'''
def CC():

    array_params = {
    'bpa': 8,
    'bpw': 8,
    'adc': 64,
    'adc_mux': 8,
    'wl': 256,
    'bl': 256,
    'offset': 128,
    'max_rpr': 128,
    }

    # 'hrs': [0.50 / 10., 0.48 / 18., 0.35 / 30.],

    arch_params1 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12],
    'lrs': [0.02, 0.04, 0.06, 0.08, 0.10], 
    'hrs': [0.03, 0.015],
    'cards': [1],
    'profile': [1],
    'rpr_alloc': ['static'],
    'thresh': [0.10, 0.25],
    }

    arch_params2 = {
    'skip': [1],
    'alloc': ['block'],
    'narray': [2 ** 12],
    'lrs': [0.02, 0.04, 0.06, 0.08], 
    'hrs': [0.03, 0.015],
    'cards': [0],
    'profile': [1],
    'rpr_alloc': ['static'],
    'thresh': [0.25],
    }

    arch_params1 = perms(arch_params1)
    arch_params2 = perms(arch_params2)
    arch_params = arch_params1 + arch_params2
    return array_params, arch_params
'''
#######################################################







