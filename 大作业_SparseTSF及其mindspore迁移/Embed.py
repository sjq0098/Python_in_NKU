import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import math


class PositionalEmbedding(nn.Cell):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = Tensor(np.zeros((max_len, d_model)).astype(np.float32))
        position = Tensor(np.arange(0, max_len).astype(np.float32).reshape(-1, 1))
        div_term = Tensor((np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model)).exp())

        pe[:, 0::2] = P.Sin()(position * div_term)
        pe[:, 1::2] = P.Cos()(position * div_term)

        pe = pe.unsqueeze(0)
        self.pe = mindspore.Parameter(pe, requires_grad=False)

    def construct(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, pad_mode='pad', padding=padding, has_bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.get_parameters():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def construct(self, x):
        x = self.tokenConv(x.swapaxes(1, 2)).swapaxes(1, 2)
        return x


class FixedEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = Tensor(np.zeros((c_in, d_model)).astype(np.float32))
        position = Tensor(np.arange(0, c_in).astype(np.float32).reshape(-1, 1))
        div_term = Tensor((np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model)).exp())

        w[:, 0::2] = P.Sin()(position * div_term)
        w[:, 1::2] = P.Cos()(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.embedding_table.set_data(Tensor(w, mindspore.float32))
        self.emb.embedding_table.requires_grad = False

    def construct(self, x):
        return self.emb(x)


class TemporalEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def construct(self, x):
        x = x.astype(mindspore.int32)

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Dense(d_inp, d_model, has_bias=False)

    def construct(self, x):
        return self.embed(x)


class DataEmbedding(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)