import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class moving_avg1(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride, c_in):
        super(moving_avg1, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        ##原始为3
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv = nn.Conv1d(in_channels=c_in - kernel_size + 1, out_channels=c_in,
                              kernel_size=3, padding=padding, padding_mode='circular', bias=False).to("cuda")

    def forward(self, x):
        # padding on the both ends of time series
        # front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x

## origin Decomp
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

## MSDecomp
class msdecomp(nn.Module):

    def __init__(self, kernel_size, c_in=96, c_out=96):
        super(msdecomp, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.dimchange_Conv = nn.Conv1d(in_channels=c_in, out_channels=c_out,
                                        kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        moving_mean = self.dimchange_Conv(moving_mean)
        res = self.dimchange_Conv(res)
        return res, moving_mean



class MSDEncoderLayer(nn.Module):
    """
    MSDEncoderLayer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(MSDEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = msdecomp(moving_avg)
            self.decomp2 = msdecomp(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.layer = torch.nn.Linear(1, 2)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x_trend_array = []

        x = x + self.dropout(new_x)
        x, x_trend1 = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        res, x_trend2 = self.decomp2(x + y)

        ## weight sum
        x_trend_array.append(x_trend1.unsqueeze(-1))
        x_trend_array.append(x_trend2.unsqueeze(-1))
        x_trend_array = torch.cat(x_trend_array, dim=-1)
        x_trend = torch.sum(x_trend_array * nn.Softmax(-1)(self.layer(x_trend1.unsqueeze(-1))), dim=-1)

        return res, x_trend, attn


class MSDEncoder(nn.Module):
    """
    MSDEncoder encoder
    """

    def __init__(self, attn_layers, d_model, seq_len, pred_len, c_out,  conv_layers=None, norm_layer=None):
        super(MSDEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.layer = torch.nn.Linear(1, 2)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.dim_Conv1 = nn.Conv1d(in_channels=seq_len, out_channels=pred_len,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.dim_Conv2 = nn.Conv1d(in_channels=seq_len, out_channels=pred_len,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.dimlinear = nn.Linear(d_model, c_out)
        self.dimlinear1 = nn.Linear(d_model, c_out)
        self.dimlinear2 = nn.Linear(d_model, c_out)

    def forward(self, x, attn_mask=None, weight_scale=0.01):
        attns = []
        res_trend = torch.zeros_like(x)
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, x_trend, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                x_restrend = x_trend + x_restrend
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, x_trend, attn = attn_layer(x, attn_mask=attn_mask)
                res_trend = weight_scale*(x_trend + res_trend)


        if self.norm is not None:
            x = self.norm(x)
        x = self.dimlinear(self.dim_Conv1(x))
        res_trend = self.dimlinear2(self.dim_Conv2(res_trend))

        return x, res_trend, attns