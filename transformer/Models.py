''' Define the Transformer model '''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Layers import EncoderLayer
from transformer.Modules import SelfAttention, ScaledDotProductAttention
from transformer.SubLayers import MATimeAttention

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_subsequent_mask_by_len(seq, len=5):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    for i in range(subsequent_mask.size(1)):
        for j in range(i+len, subsequent_mask.size(0)):
            subsequent_mask[j][i] = 1
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class PositionalEncoding(nn.Module):
    '''Position Encoding'''

    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[-x.size(1):, :]

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            ):

        super().__init__()
        feature_size = d_model

        self.pos_encoder = PositionalEncoding(feature_size, max_len=len_max_seq+5)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_subsequent_mask(src_seq)

        # -- Forward
        enc_output = self.pos_encoder(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            len_max_seq, feature=5,
            d_word_vec=16, d_model=16, d_inner=128,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        self.linear = nn.Linear(feature, d_word_vec)

        self.encoder = Encoder(
            len_max_seq=len_max_seq,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model*len_max_seq, 3, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq):
        src_seq = self.linear(src_seq)

        batch, stock, seq_len, dim = src_seq.size()
        src_seq = torch.reshape(src_seq, (-1, seq_len, dim))

        enc_output, *_ = self.encoder(src_seq)

        enc_output = enc_output.contiguous().view(-1, seq_len*dim)
        seq_logit = self.tgt_word_prj(enc_output) * self.x_logit_scale
        return seq_logit

class MATransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            len_max_seq, nums,
            t1, t2, t3,
            feature=5,
            d_word_vec=16, d_model=16, d_inner=128,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.linear = nn.Linear(feature, d_word_vec)

        self.encoder = Encoder(
            len_max_seq=len_max_seq,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.slf_attn = MATimeAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, nums=nums)

        self.tgt_word_prj = nn.Linear(d_model, 3, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq):
        src_seq = self.linear(src_seq)

        batch, stock, seq_len, dim = src_seq.size()
        src_seq = torch.reshape(src_seq, (-1, seq_len, dim))

        enc_output, *_ = self.encoder(src_seq)

        slf_attn_mask1 = get_subsequent_mask_by_len(enc_output, len=self.t1)
        slf_attn_mask2 = get_subsequent_mask_by_len(enc_output, len=self.t2)
        slf_attn_mask3 = get_subsequent_mask_by_len(enc_output, len=self.t3)
        slf_attn_mask4 = get_subsequent_mask(enc_output)

        enc_output = self.slf_attn(enc_output, enc_output, enc_output,
                                   mask1=slf_attn_mask1,
                                   mask2=slf_attn_mask2,
                                   mask3=None,
                                   mask4=None)

        enc_output = enc_output.contiguous().view(-1, dim)
        seq_logit = self.tgt_word_prj(enc_output) * self.x_logit_scale
        return seq_logit