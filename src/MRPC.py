
import numpy as np
import tensorflow as tf

from layers import *
from conv import *
from block import *
from model import *
from dense import *
from bert import *

################

def load_inputs(num_example):
    dataset = np.load('../MRPC.npy', allow_pickle=True).item()
    #########################################################################
    input_ids = dataset['input_ids'].reshape(408, 128)
    token_type_ids = dataset['token_type_ids'].reshape(408, 128)
    masks = dataset['attention_mask'].reshape(408, 128)
    labels = dataset['labels']
    #########################################################################
    xs = (input_ids[0:num_example], token_type_ids[0:num_example], masks[0:num_example])
    ys = labels[0:num_example]
    #########################################################################
    return xs, ys

################

def load_mrpc(num_example, array_params):
    model = Bert(array_params)
    x, y = load_inputs(num_example)
    return model, x, y

################























