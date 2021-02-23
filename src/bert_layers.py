
import numpy as np
import math
from scipy.special import erf

#########################################################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# https://arxiv.org/pdf/1606.08415.pdf
def gelu(x):
    return x * 0.5 * (1. + erf(x / np.sqrt(2.)))

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
    def __init__(self, weights):
        self.query = LinearLayer(weights['q'])
        self.key = LinearLayer(weights['k'])
        self.value = LinearLayer(weights['v'])
        self.attention = LinearLayer(weights['a'])
        self.norm1 = NormLayer(weights['a']['norm'])

        self.hidden = LinearLayer(weights['h'])
        self.output = LinearLayer(weights['o'])
        self.norm2 = NormLayer(weights['o']['norm'])
    
    def forward(self, x, mask):
        batch = np.shape(x)[0]
        q = self.query.forward(x).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        k = self.key.forward(x).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        v = self.value.forward(x).reshape(batch, 128, 12, 64).transpose(0, 2, 1, 3)
        
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(64)
        attention_scores = attention_scores + mask
        attention_probs = softmax(attention_scores)

        context_layer = np.matmul(attention_probs, v)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch, 128, 768)

        a = self.attention.forward(context_layer)
        a = self.norm1.forward(a + x)
        
        h = self.hidden.forward(a)
        h = gelu(h)
        
        o = self.output.forward(h)
        o = self.norm2.forward(o + a)
        
        return o

class LinearLayer():
    def __init__(self, weights):
        self.w = weights['w']
        self.b = weights['b']
    def forward(self, x):
        #####################################
        '''
        qx = quantize_and_dequantize(x)
        qw = quantize_and_dequantize(self.w)
        return qx @ qw + self.b
        '''
        #####################################
        qx, sx = quantize(x)
        qw, sw = quantize(self.w)
        qy = qx @ qw
        y = (sx * sw) * qy + self.b
        #####################################
        return y
        
class NormLayer():
    def __init__(self, weights):
        self.w = weights['w']
        self.b = weights['b']
        self.eps = weights['eps']
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
        
    def forward(self, ids, tok):
        inputs_embeds = self.word[ids]
        token_type_embeddings = self.tok[tok]
        
        position_ids = np.arange(128, dtype=int).reshape(1, -1)
        position_embeddings = self.pos[position_ids]

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.norm.forward(embeddings)
        
        return embeddings
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
