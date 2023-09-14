# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from seq2seq import *

class wr_xformer(nn.Module):
    def __init__(self, args, device='cpu', num_nodes=207, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(wr_xformer, self).__init__()
        self.xformer = Informer(args)
      
    def forward(self, x, y):
        x_enc = x[:, :1, :, :]

        x_mark_enc = x[:, 1:, 0, :]  # b,d,t
        x_dec = torch.cat([x_enc, torch.zeros_like(x_enc)], dim=3)
        x_mark_dec = torch.cat([x[:, 1:, 0, :], y[:, 1:, 0, :]], dim=2)

        b, d, n, t = x_enc.shape

        x_enc = x_enc.permute(0, 3, 1, 2).reshape(b, t, -1)  # b t d n
        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        x_dec = x_dec.permute(0, 3, 1, 2).reshape(b, t*2, -1)
        x_mark_dec = x_mark_dec.permute(0, 2, 1)


        dec_out = self.xformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # dec_out = self.xformer(x_enc, None, x_dec, None)

        dec_out = dec_out.unsqueeze(-1)


        return dec_out  # b, t, n, d1 torch.Size([128, 12, 325, 1])



class wr_xformer_single(nn.Module):
    def __init__(self, args, device='cpu', num_nodes=207, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(wr_xformer_single, self).__init__()
        self.xformer = Informer_t_and_s(args)


    def forward(self, x, y):  # bdnt

        b, d, n, t = x.shape
        x = x.permute(0, 2, 3, 1)  # bntd
        y = y.permute(0, 2, 3, 1)

        x_enc = x[:, :, :, :1]
        x_mark_enc = x.reshape((b*n), t, d)[:, :, 1:]
        x_dec = torch.cat([x_enc, torch.zeros_like(x_enc)], dim=2)
        x_mark_dec = torch.cat([x_mark_enc, y.reshape((b*n), t, d)[:, :, 1:]], dim=1)

        dec_out = self.xformer(x_enc, x_mark_enc, x_dec, x_mark_dec)  # (bn)td
        # btd -> btd


        return dec_out  #





