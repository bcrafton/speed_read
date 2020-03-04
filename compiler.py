
import math
import numpy as np
from layers import Conv
from layers import Dense
from defines import *

def model2inst(model):
    insts = []
    offset = 0
    nlayers = len(model)
    for l in range(nlayers):
        rbank = l
        wbank = l + 1
        wformat = model[l].pformat() if (l == (nlayers - 1)) else model[l + 1].rformat()
        if wformat in [FORMAT_VEC_32, FORMAT_VEC_64, FORMAT_VEC_128]: wformat = FORMAT_VEC
        assert (wformat not in [FORMAT_VEC_32, FORMAT_VEC_64, FORMAT_VEC_128])
        inst = model[l].matrix_inst(rbank=rbank, wbank=wbank, offset=offset, wformat=wformat)
        insts.append(inst)
        offset += model[l].weight_rows()
    return insts

def inst2code(insts, fmt='bin'):

    def inst2bin(inst, fmt='bin'):
        bin_inst = 0
        bin_inst = (bin_inst << DIMS_BITS)          | inst['dims']
        bin_inst = (bin_inst << READ_ROW_BITS)      | inst['rrow']
        bin_inst = (bin_inst << NVINST_BITS)        | inst['nvinst']
        bin_inst = (bin_inst << WFORMAT_BITS)       | inst['wformat']
        bin_inst = (bin_inst << PFORMAT_BITS)       | inst['pformat']
        bin_inst = (bin_inst << RFORMAT_BITS)       | inst['rformat'] 
        bin_inst = (bin_inst << OFFSET_BITS)        | inst['offset']
        bin_inst = (bin_inst << WBANK_BITS)         | inst['wbank']
        bin_inst = (bin_inst << RBANK_BITS)         | inst['rbank']
        bin_inst = (bin_inst << OPCODE_BITS)        | inst['opcode']
        if   fmt == 'bin': bin_inst = format(bin_inst, 'b').zfill(INST_BITS)
        elif fmt == 'hex': bin_inst = format(bin_inst, 'x').zfill((INST_BITS // 4) + (INST_BITS % 4))
        else:              assert (False)
        return bin_inst

    code = ''
    for i in range(128):
        if (i < len(insts)): code += inst2bin(insts[i], fmt) + '\n'
        else:                code += nop(size=64, fmt=fmt) + '\n'

    return code

#########################

def model2vinst(model):
    insts = []
    vector_offset = 0
    nlayers = len(model)
    for l in range(nlayers):
        inst = model[l].vector_inst(vector_offset=vector_offset)
        insts.append(inst)
        vector_offset += model[l].bias_rows()
        vector_offset += model[l].quant_rows()
    return insts

def vinst2code(insts, fmt='bin'):

    def vinst2bin(inst, fmt='bin'):
        bin_inst = 0
        bin_inst = (bin_inst << VECTOR_DST_ADDR_BITS)  | inst['dst_addr']
        bin_inst = (bin_inst << VECTOR_SRC2_ADDR_BITS) | inst['src2_addr']
        bin_inst = (bin_inst << VECTOR_SRC1_ADDR_BITS) | inst['src1_addr']
        bin_inst = (bin_inst << VECTOR_DST_BITS)       | inst['dst'] 
        bin_inst = (bin_inst << VECTOR_SRC2_BITS)      | inst['src2']
        bin_inst = (bin_inst << VECTOR_SRC1_BITS)      | inst['src1']
        bin_inst = (bin_inst << VECTOR_OPCODE_BITS)    | inst['opcode']
        if   fmt == 'bin': bin_inst = format(bin_inst, 'b').zfill(VECTOR_INST_BITS)
        elif fmt == 'hex': bin_inst = format(bin_inst, 'x').zfill((VECTOR_INST_BITS // 4) + (VECTOR_INST_BITS % 4))
        else:              assert (False)
        return bin_inst

    code = ''
    for i in range(1024 // VINST_PER_INST):
        for j in range(VINST_PER_INST):
            if (i < len(insts) and (j < len(insts[i]))): code += vinst2bin(insts[i][j], fmt) + '\n'
            else:                                        code += nop(size=32, fmt=fmt) + '\n'

    return code

#########################

def compile_code(model, path):
    insts = model2inst(model)
    vinsts = model2vinst(model)

    code = inst2code(insts, fmt='bin')
    f = open("%s/code.bin" % (path), 'w')
    f.write(code)
    f.close()

    code = inst2code(insts, fmt='hex')
    f = open("%s/code.hex" % (path), 'w')
    f.write(code)
    f.close()

    vcode = vinst2code(vinsts, fmt='bin')
    f = open("%s/vcode.bin" % (path), 'w')
    f.write(vcode)
    f.close()

    vcode = vinst2code(vinsts, fmt='hex')
    f = open("%s/vcode.hex" % (path), 'w')
    f.write(vcode)
    f.close()

#########################

def toBinary(n):
    return ''.join(str(1 & int(n) >> i) for i in range(8)[::-1])

def twos_complement(val, nbit):
    sign = val < 0
    mask = (1 << nbit) - 1;
    uval = abs(val) & mask

    if (sign):
        ret = ~uval + 1
    else:
        ret = uval

    return ret

def vec2bit(vecs, nbit):
    nvec = len(vecs)
    nbits = nvec * nbit
    bits = [0] * nbits

    for v in range(nvec):
        vec = twos_complement(vecs[v], nbit)

        for b in range(nbit):
            bit = v * nbit + b
            bits[bit] = (vecs[v] >> b) & 1;
            
    assert(nbits % 32 == 0)
    nword = nbits // 32
    bit_vec = [0] * nword

    bits.reverse()

    bit_str = ''
    for b in range(0, nbits, 4): # 4 for each hex digit.
        hex_val = (bits[b] << 3) + (bits[b+1] << 2) + (bits[b+2] << 1) + bits[b+3]
        # bit_str += format(hex_val, 'x')
        bit_str += format(hex_val, 'b').zfill(4) # 4 for each hex digit
  
    return bit_str
    
#########################

def model2pcm(model):
    weights = []

    nlayers = len(model)
    for l in range(nlayers):
        weight = model[l].get_weights()
        if (weight is not None): 
            weights.append(weight)
            
    weights = np.concatenate(weights, axis=1)
    return weights

def pcm2bin(weights, path):
    npcm = len(weights)
    for p in range(npcm):
        pcm = weights[p]

        for w in range(4):
            content = ''
            nrow = len(pcm)
            for row in range(nrow):
                word = pcm[row][(w*8):((w+1)*8)]
                content += vec2bit(word, 4) + '\n'

            f = open("%s/w%d%d.bin" % (path, w+1, p+1), 'w')
            f.write(content)
            f.close()
            
def pcm2pcm(pcm, path):
    for ii in range(len(pcm)):
        assert(np.shape(pcm[ii]) == (8192, 32)) # 8192x32x4 -> 1024x1024
        
        pcm_ii = np.copy(pcm[ii])
        pcm_ii = pcm_ii + 8
        
        pcm_ii = np.reshape(pcm_ii, (8192, 32))
        pcm_ii_0 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 0), 1)
        pcm_ii_1 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 1), 1)
        pcm_ii_2 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 2), 1)
        pcm_ii_3 = np.bitwise_and(np.right_shift(pcm_ii.astype(int), 3), 1)
        pcm_ii = np.stack((pcm_ii_0, pcm_ii_1, pcm_ii_2, pcm_ii_3), axis=2)        
        assert(np.shape(pcm_ii) == (8192, 32, 4))
        
        pcm_ii = np.reshape(pcm_ii, (8, 1024, 128))
        pcm_ii = np.transpose(pcm_ii, (1, 2, 0))
        
        pcm_ii = np.reshape(pcm_ii, (1024, 1024))
        np.savetxt("%s/pcm%d.csv" % (path, ii+1), pcm_ii, fmt='%d', delimiter=" ")
            
#########################

def model2vector(model):
    vectors = []

    nlayers = len(model)
    for l in range(nlayers):
        vector = model[l].get_bias()
        vectors.append(vector)
        vector = model[l].get_quant()
        vectors.append(vector)
            
    vectors = np.concatenate(vectors, axis=0)
    return vectors

def vector2bin(vector, path):
    vector = np.reshape(vector, (-1, 16, 2))
    vector = np.transpose(vector, (1, 0, 2))
    
    nsram, nrow, nword = np.shape(vector)
    
    for s in range(nsram):
        sram = vector[s]
        content = ''
        for r in range(nrow):
            content += vec2bit(vector[s][r], 16) + '\n'

        f = open("%s/b%d.bin" % (path, s+1), 'w')
        f.write(content)
        f.close()

def vector2scan(vector, path):
    nrow, nword = np.shape(vector)
    assert(nword == 32)
    np.savetxt("%s/aram.csv" % (path), vector, fmt='%d', delimiter=" ")

#########################

def compile_pcm(x, model, path):
    pcm = model2pcm(model)
    npcm, nrow, ncol = np.shape(pcm)
    zeros = np.zeros(shape=(npcm, 8192 - nrow, ncol), dtype=int)
    pcm = np.concatenate((pcm, zeros), axis=1)
    # pcm2bin(pcm, path)
    pcm2pcm(pcm, path)

    vector = model2vector(model)
    nrow, ncol = np.shape(vector)
    zeros = np.zeros(shape=(128 - nrow, ncol), dtype=int)
    vector = np.concatenate((vector, zeros), axis=0)
    vector2bin(vector, path)
    vector2scan(vector, path)
    
    n, h, w, c = np.shape(x)
    for ii in range(n):
        np.savetxt("%s/x%d.csv" % (path, ii+1), np.reshape(x[ii], -1), fmt='%d', delimiter=" ")
    
#########################

def compile_emu(x, model, path):
    emu = {}

    nlayers = len(model)
    for l in range(nlayers):
        if (model[l].opcode() == OPCODE_CONV):
            emu[l] = {'weights': model[l].weights, 'bias': model[l].bias, 'quant': model[l].quant,
                          'op': model[l].opcode(), 'x': model[l].input_size, 
                          'dims': {'stride': model[l].stride, 'pad1': model[l].pad1, 'pad2': model[l].pad2}}
        else:
            emu[l] = {'weights': model[l].weights, 'bias': model[l].bias, 'quant': model[l].quant,
                          'op': model[l].opcode(), 'x': model[l].input_size}

    n, h, w, c = np.shape(x)
    emu['x'] = x
    emu['num_example'] = n
    emu['num_layer'] = nlayers

    np.save("%s/emu" % (path), emu)

#########################

def total_macs(model):
    total = 0
    nlayers = len(model)
    for l in range(nlayers):
        total += model[l].total_macs()
    return total

def total_pcm_scans(model):
    total = 0
    nlayers = len(model)
    for l in range(nlayers):
        total += model[l].total_pcm_scans()
    return total

def total_aram_scans(model):
    total = 0
    nlayers = len(model)
    for l in range(nlayers):
        total += model[l].total_aram_scans()
    return total

#########################





























