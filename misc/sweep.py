
import numpy as np
import copy

param_sweep = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'skip': 1,
'cards': [0, 1],
'stall': 0,
'wl': 128,
'bl': 64,
'offset': 128,
'sigma': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
'err_sigma': 0.,
}

def get_perms(param):
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
    
##################################

perms = get_perms(param_sweep)
print (perms[0])

