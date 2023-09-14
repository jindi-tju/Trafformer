import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Informer, self).__init__()
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
        super(Informer, self).__init__()
        # 这里得复制一下args
        args2 = args.copy()
        args2.seq_len = 144
        # 就取12天吧
        self.informer_short = Informer(args2)

        index = (torch.arange(l, dtype=torch.long), torch.LongTensor([i // dt for i in range(l)]))
        value = torch.ones(l, dtype=torch.float)
        M.index_put_(index, value)

        x2 = torch.einsum("bld,ls->bsd", x, M)
        self.M =
        # self.x_mark_enc_cnn = None
        self.informer_long = Informer(args2)

    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]

        dec_out1 = self.informer_short(x_enc[:, -144:, :], x_mark_enc[:, -144:, :], x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

        x_enc_mean = torch.einsum("bld,ls->bsd", x_enc, self.M)
        # x_mark_enc_long = self.x_enc_cnn(x_enc)

        dec_out2 = self.informer_long(x_enc_mean, x_mark_enc[:, -144:, :], x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)

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
