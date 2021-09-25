import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from entmax import sparsemax, entmax15, entmax_bisect
from torch.autograd import grad
import os
import argparse

def get_subsequent_mask_by_len(seq, len=5):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    for i in range(subsequent_mask.size(1)):
        for j in range(i+len, subsequent_mask.size(0)):
            subsequent_mask[j][i] = 1
    for i in range(0, subsequent_mask.size(1)-len):
        for j in range(i, i+len):
            subsequent_mask[j][i] = 1

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

a = torch.rand(1,20,3)
print(get_subsequent_mask_by_len(a, len=20))