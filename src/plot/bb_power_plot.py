
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


for skip in [0, 1]:

    ######################################

    narrays = [5472, 2 ** 13, 1.5 * 2 ** 13, 2 ** 14, 1.5 * 2 ** 14]
    layers = range(8)
    mac_per_cycles = []
    mac_per_pJs = []
    errors = []
    num_example = 1

    ys = []
    for narray in narrays:
        energy = 0.
        for layer in layers:
            query = '(skip == %d) & (profile == 1) & (alloc == "block") & (narray == %d) & (id == %d)' % (skip, narray, layer)
            samples = df.query(query)

            mac = np.array(samples['nmac'])[0]
            cycles = np.max(samples['cycle'])
            x_shape = np.array(list(samples['x_shape']))
            y_shape = np.array(list(samples['y_shape']))
            adc = np.stack(samples['adc'], axis=0)    

            ######################################

            adc_leak = cycles * narray * (128 // 8) * comp_leak_constant
            
            adc_dyn = np.sum(np.array([1,2,3,4,5,6,7,8]) * adc * comp_per_bit)
            adc_dyn += np.array(samples['ron'])[0] * rram_per_bit
            adc_dyn += np.array(samples['roff'])[0] * rram_per_bit

            ######################################

            sram_leak = cycles * narray * sram_leak_constant
            sram_dyn = (np.prod(x_shape) + np.prod(y_shape)) * 2 * sram_per_bit

            ######################################

            logic_leak = cycles * narray * logic_leak_constant
            logic_dyn = np.prod(y_shape) * logic_per_bit

            ######################################

            dff_leak = cycles * narray * dff_leak_constant
            dff_dyn = (np.prod(x_shape) + np.prod(y_shape)) * dff_per_bit

            ######################################

            ic_dyn = 0. # np.prod(x_shape) + np.prod(y_shape)

            ######################################

            dram = 0.

            ######################################

            energy += adc_dyn   #+ adc_leak
            energy += sram_dyn  #+ sram_leak
            energy += logic_dyn #+ logic_leak
            energy += dff_dyn   #+ dff_leak
            energy += ic_dyn
            energy += dram

            ######################################

        # for layer in layers:
        ys.append(energy)

        ######################################

    # for narray in narrays:
    plt.plot(narrays, ys)

    ######################################

# plt.gcf().set_size_inches(3.3, 3.)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.show()
         
####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            