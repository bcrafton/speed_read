
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

ts = ['../results.npy', '../../../speed_read_cifar10/src/results.npy']
ls = [range(20), range(8)]

for t, (test, layers) in enumerate(zip(ts, ls)):
    results = np.load(test, allow_pickle=True)
    results = ld_to_dl(results)
    df = pd.DataFrame.from_dict(results)

    ys = []
    for (skip, alloc, profile) in [(1, 'block', 1),  (1, 'layer', 1), (1, 'layer', 0), (0, 'layer', 1)]:

        ######################################

        narrays = []
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
        ys.append(2 * mac_total / energy_total / 1e12)

        ######################################

    # for (skip, alloc, profile) in [(1, 'block', 1),  (1, 'layer', 1), (1, 'layer', 0), (0, 'layer', 1)]:
    if t == 1: plt.bar(x=np.array([0, 1, 2, 3]) - 0.15, height=ys, width=0.3, color='silver')
    if t == 0: plt.bar(x=np.array([0, 1, 2, 3]) + 0.15, height=ys, width=0.3, color='black')

######################################

xticks = [0, 1, 2, 3]
plt.xticks(xticks, len(xticks) * [''])

yticks = [0, 2, 4, 6, 8]
plt.yticks(yticks, len(yticks) * [''])

plt.ylim(bottom=0.0, top=8.2)

plt.gcf().set_size_inches(4.0, 2.5)
plt.tight_layout(0.)
plt.ylim(bottom=0)
# plt.show()
  
plt.gcf().savefig('power.png', dpi=500)

####################
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
