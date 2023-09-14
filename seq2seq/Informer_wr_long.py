import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Informer_wr(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Informer_wr, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Embedding                                                   timeF
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        Attn = ProbAttention if args.enc_attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention),
                        args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads, mix=args.mix),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.out_size, bias=True)
        )
    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # enc_out.shape = [134, 144, 512]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [134, 72, 512], [None, None]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [134, 324, 512]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [134, 324, 1]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Informer_wr_long(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Informer_wr_long, self).__init__()
        self.args = args
        if args.use_gpu:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # 这里得复制一下args
        # args2 = args.copy()
        ori_seq_len = args.seq_len
        args.seq_len = 144
        # 就取12天吧
        self.informer_short = Informer_wr(args)
        self.informer_long = Informer_wr(args)

        l = ori_seq_len
        s = 144
        dt = l//144
        M = torch.zeros([l, s], dtype=torch.float32)
        index = (torch.arange(l, dtype=torch.long), torch.LongTensor([i // dt for i in range(l)]))
        value = torch.ones(l, dtype=torch.float32) / dt
        M.index_put_(index, value)
        self.M = M
        if args.use_gpu:
            self.M = self.M.to('cuda:0')
        args.seq_len = ori_seq_len

        # self.x_mark_enc_cnn = None

    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]

        dec_out1 = self.informer_short(x_enc[:, -144:, :], x_mark_enc[:, -144:, :], x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

        x_enc_mean = torch.einsum("bld,ls->bsd", x_enc, self.M)

        x_dec_long_zero = torch.ones([x_enc.shape[0], self.args.pred_len, x_enc.shape[-1]]).float().to(self.device)

        x_dec_long = torch.cat([x_enc_mean[:, -self.args.label_len:, :], x_dec_long_zero], dim=1).float()

        dec_out2 = self.informer_long(x_enc_mean, x_mark_enc[:, -144:, :], x_dec_long, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

        # dec_out2 = self.informer_long(x_enc_mean, x_mark_enc[:, -144:, :], x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

        # return dec_out2
        return dec_out1+dec_out2
        #
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)  # enc_out.shape = [134, 144, 512]
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [134, 72, 512], [None, None]
        #
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [134, 324, 512]
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [134, 324, 1]
        #
        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
