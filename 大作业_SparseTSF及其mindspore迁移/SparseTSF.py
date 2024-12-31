import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
from layers.Embed import PositionalEmbedding

class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, pad_mode="pad", padding=self.period_len // 2, has_bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.SequentialCell(
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            )

    def construct(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = ops.ReduceMean(keep_dims=True)(x, 1)
        x = (x - seq_mean).swapaxes(1, 2)

        # 1D convolution aggregation
        x = self.conv1d(x.view(-1, 1, self.seq_len)).view(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.view(-1, self.seg_num_x, self.period_len).swapaxes(1, 2)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.swapaxes(1, 2).view(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.swapaxes(1, 2) + seq_mean

        return y