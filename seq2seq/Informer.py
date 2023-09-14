import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Informer_t_and_s(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Informer_t_and_s, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Embedding                                                   timeF
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout, args)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout, args)
        Attn = ProbAttention if args.enc_attn=='prob' else FullAttention  # TODO 试试

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

        self.w = nn.Parameter(torch.randn((args.num_nodes, args.d_model), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=2))

    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        b, n, t, d = x_enc.shape
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]
        enc_out = self.enc_embedding(x_enc.reshape((b*n), t, d), x_mark_enc)  # enc_out.shape = [134, 144, 512]
        # 加node_emb
        enc_out = enc_out.reshape(b, n, t, -1) + self.w  # bntd
        enc_out = enc_out.reshape(b, n*t, -1)  # b(nt)d TODO t_and_s

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # b(nt/2)d 得看看用了多少att，是根据t算的还是根据t*n算的, 另外这个(nt/2)代表什么

        # enc_out = enc_out.reshape(b, n, -1, self.args.d_model)  # b,n,t/2,d

        dec_out = self.dec_embedding(x_dec.reshape((b*n), 2*t, d), x_mark_dec)  # [134, 324, 512]
        # 加node_emb d_ff=512, d_model=64
        dec_out = dec_out.reshape(b, n, 2*t, self.args.d_model) + self.w
        dec_out = dec_out.reshape(b, n*2*t, self.args.d_model)  # TODO t_and_s

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [134, 324, 1]
        dec_out = dec_out.reshape(b, n, 2*t, -1)
        dec_out = dec_out[:, :, -t:, :]  # bntd
        dec_out = dec_out.permute(0, 2, 1, 3)  # btnd

        # dec_out = dec_out.reshape(b, -1, self.pred_len)
        # dec_out = dec_out[:, -t*n:, :]
        # dec_out = dec_out.reshape(b, t, n, -1)  # 这里这个维度直接弄的，可能会影响加gcn

        return dec_out

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class Informer_graph(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    这个版本加入了可学习图嵌入
    """
    def __init__(self, args):
        super(Informer_graph, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Embedding                                                   timeF
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout, args)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout, args)
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

        self.w = nn.Parameter(torch.randn((args.num_nodes, args.d_model), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=2))

    # x_mark_enc 和 x_mark_dec 是3维的时间emb
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        b, n, t, d = x_enc.shape
        # x_enc=[134, 144, 12], x_mark_enc=[134, 144, 3], x_dec=[134, 324, 12], x_mark_dec=[134, 324, 3]
        enc_out = self.enc_embedding(x_enc.reshape((b*n), t, d), x_mark_enc)  # enc_out.shape = [134, 144, 512]
        # 加node_emb
        enc_out = enc_out.reshape(b, n, t, -1) + self.w
        enc_out = enc_out.reshape((b*n), t, -1)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [134, 72, 512], [None, None]

        # enc_out = enc_out.reshape(b, n, -1, self.args.d_model)  # b,n,t/2,d

        dec_out = self.dec_embedding(x_dec.reshape((b*n), 2*t, d), x_mark_dec)  # [134, 324, 512]
        # 加node_emb d_ff=512, d_model=64
        dec_out = dec_out.reshape(b, n, 2*t, self.args.d_model) + self.w
        dec_out = dec_out.reshape((b*n), 2*t, self.args.d_model)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # [134, 324, 1]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


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
                                           args.dropout, args)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout, args)
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

class InformerStack_wr(nn.Module):
    def __init__(self, args, e_layers=[3, 2, 1]):
    # def __init__(self, args, enc_in, dec_in, out_size, out_len,
    #             factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512,
    #             dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
    #             output_attention = False, distil=True, mix=True):
        super(InformerStack_wr, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.attn = args.enc_attn  # TODO attn 和下面Attn好像一样
        self.output_attention = args.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        # Attention
        Attn = ProbAttention if args.enc_attn=='prob' else FullAttention
        # Encoder
        # TODO 暂时保留该参数，这个虽然informer class中也叫e_layers，但是那个只是个数，这里是个数组
        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention),
                                    args.d_model, args.n_heads, mix=False),
                        args.d_model,
                        args.d_ff,
                        dropout=args.dropout,
                        activation=args.activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        args.d_model
                    ) for l in range(el-1)
                ] if args.distil else None,
                norm_layer=torch.nn.LayerNorm(args.d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                                args.d_model, args.n_heads, mix=args.mix),
                    AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                                args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=out_size, kernel_size=1, bias=True)
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enout_size = self.enc_embedding(x_enc, x_mark_enc)
        enout_size, attns = self.encoder(enout_size, attn_mask=enc_self_mask)

        deout_size = self.dec_embedding(x_dec, x_mark_dec)
        deout_size = self.decoder(deout_size, enout_size, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        deout_size = self.projection(deout_size)
        
        # deout_size = self.end_conv1(deout_size)
        # deout_size = self.end_conv2(deout_size.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return deout_size[:,-self.pred_len:,:], attns
        else:
            return deout_size[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, out_size, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=out_size, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, out_size, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enout_size = self.enc_embedding(x_enc, x_mark_enc)
        enout_size, attns = self.encoder(enout_size, attn_mask=enc_self_mask)

        deout_size = self.dec_embedding(x_dec, x_mark_dec)
        deout_size = self.decoder(deout_size, enout_size, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        deout_size = self.projection(deout_size)

        # deout_size = self.end_conv1(deout_size)
        # deout_size = self.end_conv2(deout_size.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return deout_size[:, -self.pred_len:, :], attns
        else:
            return deout_size[:, -self.pred_len:, :]  # [B, L, D]
