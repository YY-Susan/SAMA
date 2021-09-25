import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.SubLayers import MultiHeadAttention, MATimeAttention
from transformer.Models import get_subsequent_mask, get_subsequent_mask_by_len




class SAMA(nn.Module):
    def __init__(
            self,
            len_max_seq, nums,
            t1, t2, t3,
            feature=5, d_model=16,
            n_layers=3, n_head=4, d_k=4, d_v=4, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.linear = nn.Linear(feature, d_model)
        self.rnn = nn.LSTM(d_model,
                           d_model,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=False)
        self.slf_attn = MATimeAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, nums=nums)

        self.tgt_word_prj = nn.Linear(d_model, 3, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        if tgt_emb_prj_weight_sharing:
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq):
        src_seq = self.linear(src_seq)
        batch, stock, seq_len, dim = src_seq.size()

        src_seq = torch.reshape(src_seq, (-1, seq_len, dim))
        enc_output, _ = self.rnn(src_seq)

        slf_attn_mask1 = get_subsequent_mask_by_len(enc_output, len=self.t1)
        slf_attn_mask2 = get_subsequent_mask_by_len(enc_output, len=self.t2)
        slf_attn_mask3 = get_subsequent_mask_by_len(enc_output, len=self.t3)
        slf_attn_mask4 = get_subsequent_mask(enc_output)

        enc_output = self.slf_attn(enc_output, enc_output, enc_output,
                                   mask1=slf_attn_mask1,
                                   mask2=None,
                                   mask3=None,
                                   mask4=None)

        enc_output = enc_output.contiguous().view(-1, dim)

        seq_logit = self.tgt_word_prj(enc_output) * self.x_logit_scale
        return seq_logit



