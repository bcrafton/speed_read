
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####################

def ld_to_dl(ld):
    dl = {}

    for i, d in enumerate(ld):
        for key in d.keys():
            value = d[key]
            if i == 0:
                dl[key] = [value]
            else:
                dl[key].append(value)

    return dl

####################

results = np.load('../results.npy', allow_pickle=True)
# print (len(results))

results = ld_to_dl(results)
df = pd.DataFrame.from_dict(results)

# print (df.columns)
# print (df['layer_id'])
# print (df['id'])

####################

comp_per_bit = 45e-15; rram_per_bit = 2e-16
sram_per_bit = 64e-15
dff_per_bit  = 85e-15
logic_per_bit = 100e-15 # shift + add

comp_leak_constant  = comp_per_bit * 0.05  # 0.05 comes from 6b Flash ADC paper.
sram_leak_constant  = sram_per_bit * 0.20  # depends on size of bank (CACTI), but should really synthesize
logic_leak_constant = logic_per_bit * 0.05 # should really synthesize this + SRAM -> PrimeTime
dff_leak_constant   = dff_per_bit * 0.05   # should really synthesize this + SRAM -> PrimeTime

#####################
# NOTE: FOR QUERIES WITH STRINGS, DONT FORGET ""
# '(rpr_alloc == "%s")' NOT '(rpr_alloc == %s)'
#####################

ys = []
for (skip, alloc, profile) in [(1, 'block', 1),  (1, 'layer', 1), (1, 'layer', 0), (0, 'layer', 1)]:

    ######################################

    # TODO: VGG[1 vs 3] will look better becauseonly 3.5x over 3rd place
    # TODO: we dont want to sweep over narray ... they all end up being the same.
    narrays = []
    layers = range(8)
    mac_per_cycles = []
    mac_per_pJs = []
    errors = []
    num_example = 1

    energy_total = 0.
    mac_total = 0.

    query = '(skip == %d) & (profile == %d) & (alloc == "%s") & (narray == %d)' % (skip, profile, alloc, 2 ** 14)
    samples = df.query(query)
    cycles = np.max(samples['cycle'])
    
    for layer in layers:
        query = '(skip == %d) & (profile == %d) & (alloc == "%s") & (narray == %d) & (id == %d)' % (skip, profile, alloc, 2 ** 14, layer)
        samples = df.query(query)

        mac = np.array(samples['nmac'])[0]
        x_shape = np.array(list(samples['x_shape']))
        y_shape = np.array(list(samples['y_shape']))
        adc = np.stack(samples['adc'], axis=0)    

        ######################################

        adc_leak = cycles * 2 ** 14 * (128 // 8) * comp_leak_constant
        
        adc_dyn = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_per_bit)
        adc_dyn += np.array(samples['ron'])[0] * rram_per_bit
        adc_dyn += np.array(samples['roff'])[0] * rram_per_bit

        ######################################

        sram_leak = cycles * 2 ** 14 * sram_leak_constant
        sram_dyn = (np.prod(x_shape) + np.prod(y_shape)) * 2 * sram_per_bit

        ######################################

        logic_leak = cycles * 2 ** 14 * logic_leak_constant
        logic_dyn = np.prod(y_shape) * logic_per_bit

        ######################################

        dff_leak = cycles * 2 ** 14 * dff_leak_constant
        dff_dyn = (np.prod(x_shape) + np.prod(y_shape)) * dff_per_bit

        ######################################

        ic_dyn = 0. # np.prod(x_shape) + np.prod(y_shape)

        ######################################

        dram = 0.

        ######################################

        energy_total += adc_dyn   + adc_leak
        energy_total += sram_dyn  + sram_leak
        energy_total += logic_dyn + logic_leak
        energy_total += dff_dyn   + dff_leak
        energy_total += ic_dyn
        energy_total += dram

        mac_total += mac

        ######################################

    # for layer in layers:
    ys.append(mac_total / energy_total / 1e12)

    ######################################

# for (skip, alloc, profile) in [(1, 'block', 1),  (1, 'layer', 1), (1, 'layer', 0), (0, 'layer', 1)]:
plt.bar([0, 1, 2, 3], ys, width=0.35)

######################################

# plt.gcf().set_size_inches(3.3, 3.)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.show()
         
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
