
import numpy as np
import math
from scipy.special import erf
from linear import Linear

#########################################################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# https://arxiv.org/pdf/1606.08415.pdf
def gelu(x):
    return x * 0.5 * (1. + erf(x / np.sqrt(2.)))

def quantize_scale(x, scale):
    x = x / scale
    x = np.around(x)
    x = np.clip(x, -128, 127)
    return x

def quantize(x):
    scale = np.max(np.absolute(x)) / 127
    x = x / scale
    x = np.around(x)
    x = np.clip(x, -128, 127)
    return x, scale

def quantize_and_dequantize(x):
    scale = np.max(np.absolute(x)) / 127
    x = x / scale
    x = np.around(x)
    x = np.clip(x, -128, 127)
    x = x * scale
    return x#, scale

#########################################################################

class BertLayer():
    def __init__(self, params, weights):
        self.query = LinearLayer(params=params, weights=weights['q'])
        self.key = LinearLayer(params=params, weights=weights['k'])
        self.value = LinearLayer(params=params, weights=weights['v'])
        self.attention = LinearLayer(params=params, weights=weights['a'])
        self.norm1 = NormLayer(weights['a']['norm'])

        self.hidden = LinearLayer(params=params, weights=weights['h'])
        self.output = LinearLayer(params=params, weights=weights['o'])
        self.norm2 = NormLayer(weights['o']['norm'])

    def init(self, params, table):
        self.query.init(params, table)
        self.key.init(params, table)
        self.value.init(params, table)
        self.attention.init(params, table)
        self.hidden.init(params, table)
        self.output.init(params, table)

    def set_profile_adc(self, counts):
        self.query.set_profile_adc(counts)
        self.key.set_profile_adc(counts)
        self.value.set_profile_adc(counts)
        self.attention.set_profile_adc(counts)
        self.hidden.set_profile_adc(counts)
        self.output.set_profile_adc(counts)

    def profile_adc(self, x, mask, counters):
        batch = np.shape(x)[0]
        q = self.query.profile_adc(x, counters).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        k = self.key.profile_adc(x, counters).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        v = self.value.profile_adc(x, counters).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(64)
        attention_scores = attention_scores + mask
        attention_probs = softmax(attention_scores)

        context_layer = np.matmul(attention_probs, v)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch, 128, 768)

        a = self.attention.profile_adc(context_layer, counters)
        a = self.norm1.forward(a + x)
        
        h = self.hidden.profile_adc(a, counters)
        h = gelu(h)
        
        o = self.output.profile_adc(h, counters)
        o = self.norm2.forward(o + a)
        
        return o

    def forward(self, x, results):
        (x, mask) = x
        batch = np.shape(x)[0]
        q = self.query.forward(x, results).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        k =   self.key.forward(x, results).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        v = self.value.forward(x, results).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(64)
        attention_scores = attention_scores + mask
        attention_probs = softmax(attention_scores)

        context_layer = np.matmul(attention_probs, v)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch, 128, 768)

        a = self.attention.forward(context_layer, results)
        a = self.norm1.forward(a + x)
        
        h = self.hidden.forward(a, results)
        h = gelu(h)
        
        o = self.output.forward(h, results)
        o = self.norm2.forward(o + a)
        
        return o

class LinearLayer():
    def __init__(self, params, weights):
        self.input_size, self.output_size = np.shape(weights['w'])
        self.weights = weights
        self.linear = Linear((self.input_size, self.output_size), params, weights)
        
    def init(self, params, table):
        self.linear.init(params, table)

    def set_profile_adc(self, counts):
        self.linear.set_profile_adc(counts)

    def profile_adc(self, x, counters):
        batch, word, vec = np.shape(x)
        assert (batch == 1)
        x = np.reshape(x, (word, vec))        

        qx = quantize_scale(x, self.weights['sx'])
        b = self.weights['b']
        qy = qx @ self.weights['w']
        y = (self.weights['sx'] * self.weights['sw']) * qy + b
        
        qx += 128
        _ = self.linear.profile_adc(qx, counters)

        y = np.reshape(y, (batch, word, self.output_size))
        return y
    
    def forward(self, x, results):
        batch, word, vec = np.shape(x)
        assert (batch == 1)
        x = np.reshape(x, (word, vec))        

        qx = quantize_scale(x, self.weights['sx'])
        b = self.weights['b']
        qy = qx @ self.weights['w']
        y = (self.weights['sx'] * self.weights['sw']) * qy + b

        qx += 128
        pim = self.linear.forward(qx, results)
        pim -= np.sum(self.weights['w'], axis=0) * 128

        error = np.mean(np.absolute(qy - pim))
        assert (error == 0)

        y = np.reshape(y, (batch, word, self.output_size))
        return y
        
class NormLayer():
    def __init__(self, weights):
        self.w = weights['w']
        self.b = weights['b']
        self.eps = weights['eps']
    def init(self, params, table):
        pass
    def forward(self, x):
        mean = np.mean(x, axis=(2), keepdims=True)
        x = x - mean
        var = np.mean(np.square(x), axis=(2), keepdims=True)
        x = x / np.sqrt(var + self.eps) * self.w + self.b
        return x

class EmbedLayer():
    def __init__(self, weights):
        self.word = weights['word']
        self.tok = weights['tok']
        self.pos = weights['pos']
        self.norm = NormLayer(weights['norm'])
    def init(self, params, table):
        pass
    def forward(self, ids, tok):
        inputs_embeds = self.word[ids]
        token_type_embeddings = self.tok[tok]
        
        position_ids = np.arange(128, dtype=int).reshape(1, -1)
        position_embeddings = self.pos[position_ids]

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.norm.forward(embeddings)
        
        return embeddings

'''
class Model():
    def __init__(self):
        weights = np.load('weights.npy', allow_pickle=True).item()
        self.embed = EmbedLayer(weights['embed'])
        self.encoder = []
        for l in range(12):
            self.encoder.append(BertLayer(weights['encoder'][l]))
        self.pooler = LinearLayer(weights['pool'])
        self.classifier = LinearLayer(weights['class'])
    
    def forward(self, ids, tok, mask):
        embed = self.embed.forward(ids, tok)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        h = embed
        for l in range(12):
            h = self.encoder[l].forward(h, mask)
        h = h[:, 0]
        p = np.tanh(self.pooler.forward(h))
        o = self.classifier.forward(p)
        return o
'''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
