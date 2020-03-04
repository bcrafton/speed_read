
import math
import numpy as np
from conv_utils import conv_output_length
from transform_weights import transform
from defines import *

class Conv:
    NROW_BITS   = 8
    NCOL_BITS   = 8
    FILTER_BITS = 2
    STRIDE_BITS = 1
    PAD1_BITS = 2
    PAD2_BITS = 2

    def __init__(self, input_size, filter_size, stride, pad1, pad2, weights=None):
        self.input_size = input_size
        self.h, self.w, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.stride = stride
        self.pad1 = pad1
        self.pad2 = pad2
        
        self.y_h = (self.h - self.fh + self.stride + self.pad1 + self.pad2) / self.stride
        self.y_w = (self.w - self.fw + self.stride + self.pad1 + self.pad2) / self.stride
        # print (self.y_h, self.y_w)
        
        if (self.fh == 1): 
            assert((self.stride==1) and (self.pad1==0) and (self.pad2==0))

        if weights == None:
            values = np.array(range(-6, 8))
            self.weights = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            self.bias = np.zeros(shape=self.fn).astype(int) # np.random.choice(a=values, size=self.fn, replace=True).astype(int)
            # TODO: make me a lut function based based on input size
            self.quant = np.ones(shape=self.fn).astype(int) * 200
        else:
            self.weights, self.bias, self.quant = weights
            assert(np.shape(self.weights) == self.filter_size)
            assert(np.shape(self.bias) == (self.fn,))
            assert(np.shape(self.quant) == ())

            self.quant = np.ones(shape=self.fn) * self.quant

            self.weights = self.weights.astype(int)
            self.bias = self.bias.astype(int)
            self.quant = self.quant.astype(int)

    def rformat(self):    
        if (self.c <= 4):     return FORMAT_4
        elif (self.c <= 8):   return FORMAT_8
        elif (self.c <= 16):  return FORMAT_16
        elif (self.c <= 32):  return FORMAT_32
        elif (self.c <= 64):  return FORMAT_64
        elif (self.c <= 128): return FORMAT_128
        else: assert (False)
        
    def pformat(self):
        if   (self.fn <= 4):   return FORMAT_4
        elif (self.fn <= 8):   return FORMAT_8
        elif (self.fn <= 16):  return FORMAT_16
        elif (self.fn <= 32):  return FORMAT_32
        elif (self.fn <= 64):  return FORMAT_64
        elif (self.fn <= 128): return FORMAT_128
        else: assert (False)
    
    def opcode(self):
        return OPCODE_CONV
        
    def dims(self):
        dim = 0
        dim = (dim << Conv.PAD2_BITS)    | self.pad2
        dim = (dim << Conv.PAD1_BITS)    | self.pad1
        dim = (dim << Conv.STRIDE_BITS)  | (self.stride - 1)

        dim = (dim << Conv.FILTER_BITS) | (self.fh - 1)
        dim = (dim << Conv.NCOL_BITS)   | self.w
        dim = (dim << Conv.NROW_BITS)   | self.h
        return dim
        
    def get_weights(self): 
        return transform(self.weights)
        
    def get_bias(self):
        return np.reshape(self.bias, (-1, 32))
        
    def get_quant(self):
        q_shift = np.right_shift(self.quant, self.rrow())
        return np.reshape(q_shift, (-1, 32))
        
    def weight_rows(self): 
        # what about rounding ??
        return np.ceil((self.fh * self.fw * self.fc * self.fn) / (8 * 4 * 32)).astype(int) # 8 ROW * 4 BLOCK * 32 MAC each 
        
    def bias_rows(self):
        return (self.fn // 32)
        
    def quant_rows(self):
        return (self.fn // 32)
        
    def rrow(self):
        return 0

    def matrix_inst(self, rbank, wbank, offset, wformat):
        inst = {}
        inst['rbank'] = rbank
        inst['wbank'] = wbank
        inst['offset'] = offset
        inst['wformat'] = wformat

        inst['opcode'] = self.opcode()
        inst['rformat'] = self.rformat()
        inst['pformat'] = self.pformat()
        inst['nvinst'] = (self.fn // 32) * 3 - 1
        inst['rrow'] = self.rrow()
        inst['dims'] = self.dims()
        return inst

    def vector_inst(self, vector_offset):
        insts = []
    
        nblock = self.fn // 32
        for block in range(nblock):
            bias = {}
            bias['opcode'] = VECTOR_OPCODE_ADD
            bias['src1'] = VECTOR_SRC1_ACCUM
            bias['src2'] = VECTOR_SRC2_ARAM
            bias['dst'] = VECTOR_DST_VACCUM
            bias['src1_addr'] = block
            bias['src2_addr'] = block + vector_offset
            bias['dst_addr'] = 0 # dont care.
            insts.append(bias)

            relu = {}
            relu['opcode'] = VECTOR_OPCODE_RELU
            relu['src1'] = VECTOR_SRC1_VACCUM
            relu['src2'] = 0 # dont care.
            relu['dst'] = VECTOR_DST_VACCUM
            relu['src1_addr'] = 0 # dont care.
            relu['src2_addr'] = 0 # dont care.
            relu['dst_addr'] = 0 # dont care.
            insts.append(relu)
            
            quant = {}
            quant['opcode'] = VECTOR_OPCODE_QUANT
            quant['src1'] = VECTOR_SRC1_VACCUM
            quant['src2'] = VECTOR_SRC2_ARAM
            quant['dst'] = VECTOR_DST_FRAM
            quant['src1_addr'] = block
            quant['src2_addr'] = block + vector_offset + self.bias_rows()
            quant['dst_addr'] = block
            insts.append(quant)
            
        return insts
        
    def total_macs(self):
        mac_per_patch = (self.fh * self.fw * self.fc * self.fn) 
        npatch = (self.y_h * self.y_w)
        return npatch * max(256 * 4, mac_per_patch) # min = 4 BLOCK * 32 MAC each * 8 ROW

    def total_pcm_scans(self):
        scans = (self.fh * self.fw * self.fc * self.fn) 
        return scans

    def total_aram_scans(self):
        scans = self.fn * 2
        return scans

#########################

class Dense:
    NROW_BITS = 10
    NCOL_BITS = 10
    
    def __init__(self, size, weights=None):        
        self.size = size
        self.input_size, self.output_size = self.size
        assert((self.output_size == 32) or (self.output_size == 64) or (self.output_size == 128))

        if weights == None:
            values = np.array(range(-1, 4))
            self.weights = np.random.choice(a=values, size=self.size, replace=True).astype(int)
            
            # np.random.choice(a=values, size=self.output_size, replace=True).astype(int)
            self.bias = np.zeros(shape=self.output_size).astype(int) 
            
            # make lut function based on input size
            self.quant = np.ones(shape=self.output_size).astype(int) * 200
        else:
            self.weights, self.bias, self.quant = weights
            assert(np.shape(self.weights) == self.size)
            assert(np.shape(self.bias) == (self.output_size,))
            assert(np.shape(self.quant) == ())
            
            self.quant = np.ones(shape=self.output_size) * self.quant

            self.weights = self.weights.astype(int)
            self.bias = self.bias.astype(int)
            self.quant = self.quant.astype(int)

    def rformat(self):
        '''
        if   (self.input_size <= 32):  return FORMAT_32
        elif (self.input_size <= 64):  return FORMAT_64
        elif (self.input_size <= 128): return FORMAT_128
        else: assert (False)
        '''
        return FORMAT_VEC

    def pformat(self):
        if   (self.output_size <= 32):  return FORMAT_VEC_32
        elif (self.output_size <= 64):  return FORMAT_VEC_64
        elif (self.output_size <= 128): return FORMAT_VEC_128
        else: assert (False)
    
    def opcode(self):
        return OPCODE_DOT
        
    def dims(self):
        dim = 0
        dim = (dim << Dense.NCOL_BITS) | self.output_size
        dim = (dim << Dense.NROW_BITS) | self.input_size
        return dim
        
    '''
    def get_weights(self): 
        w = np.copy(self.weights)

        num_row, num_col = np.shape(w)
        assert(num_col == 128)

        w = np.reshape(w, (num_row, 4, 32))
        w = np.transpose(w, (1, 0, 2))
        return w
    '''
    def get_weights(self): 
        return transform(self.weights)
    
    def get_bias(self):
        return np.reshape(self.bias, (self.output_size // 32, 32))

    def get_quant(self):
        return np.reshape(self.quant, (self.output_size // 32, 32))

    def weight_rows(self):
        # what about rounding ??
        return self.input_size // 8
        
    def bias_rows(self):
        return self.output_size // 32
        
    def quant_rows(self):
        return self.output_size // 32
        
    def rrow(self):
        return 0

    def matrix_inst(self, rbank, wbank, offset, wformat):
        inst = {}
        inst['rbank'] = rbank
        inst['wbank'] = wbank
        inst['offset'] = offset
        inst['wformat'] = wformat

        inst['opcode'] = self.opcode()
        inst['rformat'] = self.rformat()
        inst['pformat'] = self.pformat()
        inst['nvinst'] = 2 * (self.output_size // 32) - 1
        inst['rrow'] = self.rrow()
        inst['dims'] = self.dims()
        return inst

    def vector_inst(self, vector_offset):
        insts = []
    
        nblock = 8
        for block in range(nblock):
            bias = {}
            bias['opcode'] = VECTOR_OPCODE_ADD
            bias['src1'] = VECTOR_SRC1_ACCUM
            bias['src2'] = VECTOR_SRC2_ARAM
            bias['dst'] = VECTOR_DST_VACCUM
            bias['src1_addr'] = block
            bias['src2_addr'] = block + vector_offset
            bias['dst_addr'] = 0 # dont care.
            insts.append(bias)
            '''
            relu = {}
            relu['opcode'] = VECTOR_OPCODE_RELU
            relu['src1'] = VECTOR_SRC1_VACCUM
            relu['src2'] = 0 # dont care.
            relu['dst'] = VECTOR_DST_VACCUM
            relu['src1_addr'] = 0 # dont care.
            relu['src2_addr'] = 0 # dont care.
            relu['dst_addr'] = 0 # dont care.
            insts.append(relu)
            '''
            quant = {}
            quant['opcode'] = VECTOR_OPCODE_QUANT
            quant['src1'] = VECTOR_SRC1_VACCUM
            quant['src2'] = VECTOR_SRC2_ARAM
            quant['dst'] = VECTOR_DST_FRAM
            quant['src1_addr'] = block
            quant['src2_addr'] = block + vector_offset + self.bias_rows()
            quant['dst_addr'] = block
            insts.append(quant)

        return insts
        
    def total_macs(self):
        return self.input_size * self.output_size 

    def total_pcm_scans(self):
        return self.input_size * self.output_size

    def total_aram_scans(self):
        return self.output_size * 2 

#########################
        
