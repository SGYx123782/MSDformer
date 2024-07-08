import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embeding import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_inverted

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.MSDformer_EncDec import MSDEncoder, MSDEncoderLayer, my_Layernorm, series_decomp, msdecomp
import math
import numpy as np

from layers.Normalize import Normalize


class MSDformer(nn.Module):
    """
    Myformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(MSDformer, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = msdecomp(kernel_size, self.seq_len, self.label_len + self.pred_len)
        else:
            self.decomp = series_decomp(kernel_size)

        self.revin_layer = Normalize(num_features=configs.enc_in, affine=False, subtract_last=False)
        self.revin_layer1 = Normalize(num_features=configs.enc_in, affine=False, subtract_last=False)

        self.autoregressive = nn.Linear(configs.d_model, configs.c_out)
        self.device = "cuda:0"

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = MSDEncoder(
            [
                MSDEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            configs.d_model,
            configs.seq_len,
            configs.label_len + configs.pred_len,
            configs.c_out,
            norm_layer=my_Layernorm(configs.d_model)
        )

        self.w1 = nn.Parameter(torch.ones(configs.batch_size, self.pred_len + self.label_len, configs.c_out), requires_grad= True,)
        self.w2 = nn.Parameter(torch.ones(configs.batch_size, self.pred_len + self.label_len, configs.c_out),
                              requires_grad=True, )
        self.w3 = nn.Parameter(torch.ones(configs.batch_size, self.pred_len + self.label_len, configs.c_out),
                              requires_grad=True, )
        self.w4 = nn.Parameter(torch.zeros(configs.batch_size, self.pred_len + self.label_len, configs.c_out),
                              requires_grad=True, )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc = self.revin_layer(x_enc, 'norm')
        x_dec = self.revin_layer1(x_dec, 'norm')

        seasonal_init, trend_init = self.decomp(x_enc)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, res_trend, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # auto
        dec_out = seasonal_init + x_dec

        dec_out = self.dec_embedding(dec_out, x_mark_dec)

        dec_out = self.autoregressive(dec_out)

        # final
        dec_out = self.w1 * dec_out + self.w2 * enc_out + self.w3 * trend_init + self.w4 * res_trend

        dec_out = self.revin_layer(dec_out, "denorm")

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :] # [B, L, D]

    def name(self):
        return self.__class__.__name__
