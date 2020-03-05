
#########################

FORMAT_VEC = 0
FORMAT_4 = 1
FORMAT_8 = 2
FORMAT_16 = 3
FORMAT_32 = 4
FORMAT_64 = 5
FORMAT_128 = 6

FORMAT_VEC_32 = 7
FORMAT_VEC_64 = 8
FORMAT_VEC_128 = 9

#########################

OPCODE_CONV  = 1
OPCODE_DOT = 2

#########################

VECTOR_OPCODE_NOP = 0
VECTOR_OPCODE_ADD = 1
VECTOR_OPCODE_MULT = 2
VECTOR_OPCODE_RELU = 3
VECTOR_OPCODE_DIV = 4
VECTOR_OPCODE_QUANT = 5
VECTOR_OPCODE_SIG = 6
VECTOR_OPCODE_TANH = 7

VECTOR_SRC1_ACCUM = 0
VECTOR_SRC1_VACCUM = 1

VECTOR_SRC2_ACCUM = 0
VECTOR_SRC2_ARAM = 1

VECTOR_DST_VACCUM = 0
VECTOR_DST_FRAM = 1
VECTOR_DST_ARAM = 2

#########################

INST_BITS = 64
FORMAT_BITS = 4

OPCODE_BITS         = 2
OPCODE_CONV         = 1
OPCODE_DOT          = 2

RBANK_BITS          = 4
WBANK_BITS          = 4

OFFSET_BITS         = 10

RFORMAT_BITS        = FORMAT_BITS
PFORMAT_BITS        = FORMAT_BITS
WFORMAT_BITS        = FORMAT_BITS

NVINST_BITS         = 6

READ_ROW_BITS       = 2

DIMS_BITS           = INST_BITS - (OPCODE_BITS + (RBANK_BITS + WBANK_BITS) + OFFSET_BITS + (RFORMAT_BITS + PFORMAT_BITS + WFORMAT_BITS) + NVINST_BITS + READ_ROW_BITS) 

#########################

VECTOR_INST_BITS = 32

VECTOR_OPCODE_BITS = 3

VECTOR_SRC1_BITS = 1
VECTOR_SRC2_BITS = 1
VECTOR_DST_BITS = 2

VECTOR_SRC1_ADDR_BITS = 7
VECTOR_SRC2_ADDR_BITS = 7
VECTOR_DST_ADDR_BITS = 7

#########################

VINST_PER_INST = 1 << NVINST_BITS

#########################

def nop(size=64, fmt='bin'):
    if (fmt == 'bin'):
        return '0' * size
    elif (fmt == 'hex'):
        return '0' * (size // 4)
    else:
        return False

#########################











