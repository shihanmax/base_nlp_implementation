#!/usr/bin/env python
# coding: utf-8

# In[79]:


import copy
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[59]:


class EncoderDecoder(nn.Module):
    
    """标准encoder-decoder架构"""
    
    def __init__(self, encoder, decoder, src_embd, tgt_embd, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd  # 源序列embedding
        self.tgt_embd = tgt_embd  # 目标序列embedding
        self.generator = generator  # 生成目标单词的概率
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """接收，处理原序列、目标序列以及他们的mask"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embd(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embd(tgt), memory, src_mask, tgt_mask)


# In[60]:


class Generator(nn.Module):
    """定义linear+softmax生成"""
    def __init__(self, d_module, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_module, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# In[61]:


def clones(module, N):
    """将module复制N份"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[62]:


class LayerNorm(nn.Module):
    """构造一个layernorm模块"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.parameter(torch.ones(features))
        self.b_2 = nn.parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        """norm"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    


# In[63]:


class Encoder(nn.Module):
    """N层堆叠的Encoder"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """每层layer依次通过输入序列与mask"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[64]:


class SublayerConnection(nn.Module):
    """Add + Norm"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """add norm"""
        return x + self.dropout(sublayer(self.norm(x)))


# In[65]:


class EncoderLayer(nn.Module):
    """encoder分为两层：self.attn + feedforward"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        """Self-attn + feedforward"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# In[66]:


class Decoder(nn.Module):
    """带mask的通用decoder"""
    def __init__(self, layer, N):
        super(Decoder, self).__init()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, scr_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, scr_mask, tgt_mask)
        return self.morm(x)


# In[67]:


class DecoderLayer(nn.Module):
    """self-attn + scr-attn + feedforward"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(Decoder, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout))
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """将decoder的三个sublayer串联"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[68]:


def subsequent_mask(size):
    """mask后续的位置，返回[size, size]的tensor，对角线及左下角全是1，其余为0"""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


# In[69]:


def attention(query, key, value, mask=None, dropout=None):
    """计算attention即点乘V"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


# In[70]:


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        """实现multiheadattention
            输入的q k v的形状为[batch, L, d_model]
            输出的x形状同上
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        
        # [batch, L, d_model] -> [batch, h, L, d_model/h]
        query, key, value = (
            [l(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]
        )
        
        # 计算注意力attention，得到attn*v与attn
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
    
        return self.linears[-1](x)


# In[71]:


class PositionwizeFeedForward(nn.Module):
    """实现FFN"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwizeFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# In[72]:


class Embeddings(nn.Module):
    
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  # embedding维度
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# In[83]:


class PositionalEncoding(nn.Module):
    """实现PE"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# In[74]:


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """根据超参，构建模型"""
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwizeFeedForward(d_model, d_ff, dropout)
    
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(cttn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # 初始化参数
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# In[75]:


batch_size = 16
L = 50  # 序列长度
d_model = 512  # 词向量维度
h = 8
x = torch.randn(batch_size, L, d_model)  # 生成测试序列x
x.size()


# In[76]:


# 测试MultiHeadAttention
obj = MultiHeadAttention(8, 512)
q = torch.randn(2, 10, 512)  # 输入序列
line_net = clones(nn.Linear(512, 512), 4)

q, k, v = [l(x).view(2, -1, 8, 64).transpose(1, 2) for l, x in zip(line_net, (q, q, q))]
print(k.size(), k.transpose(-2, -1).size())
d_k = d_model // h
print("dk: ", d_k)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
print("scores: ", scores.size())
attn = F.softmax(scores, dim=-1)
print("attn_size: ", attn.size())
rx = torch.matmul(attn, v)
print(rx.size)
out = rx.transpose(1, 2).contiguous().view(2, -1, 8 * 64)
print("out size: ", out.size())


# In[84]:


plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim:{}".format(p) for p in [4, 5, 6, 7]])


# In[85]:


class Batch(object):
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).updqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self,trg, pad)
            self.n_tokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).upsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        )
        
        return tgt_mask


# In[ ]:


def run_epoch(data_iter, model, loss_compute, device):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        src_mask= batch.src_mask.to(device)
        trg_mask = batch.trg_mask.to(device)
        trg_y = batch.trg_y.to(device)
        n_tokens = batch.n_tokens.to(device)
        
        out = model.forward(src, trg, src_mask, trg_mask)
        loss = loss_compute(out, trg_y, n_tokens)
        
        total_loss += loss.detach().cpu().numpy()
        total_tokens += n_tokens.cpu().numpy()
        tokens += n_tokens.cpu().numpy()
        
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch step:{}  Loss:{}  Tokens per sec:{}".format(
                i, loss.detach().cpu().numpy() / n_tokens.cpu().numpy(), tokens / elapsed
            ))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

