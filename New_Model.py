import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyModule(nn.Module):
    def __init__(self, seg_num_x, seg_num_y, freq_len, sparse_k=4):
        super().__init__()
        self.sparse_k = sparse_k
        self.freq_len = freq_len
        # 可学习的频率掩码
        self.freq_mask = nn.Parameter(torch.randn(freq_len) / (freq_len ** 0.5))
        # 频率重建的线性层
        self.reconstruct = nn.Linear(freq_len, seg_num_y, bias=False)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        # 应用FFT将数据转换到频域
        freq = torch.fft.rfft(x, dim=1)
        freq_amp = freq.abs()  # 频率分量的幅度
        # 应用可学习掩码
        masked_freq = freq_amp * self.freq_mask
        # 逆FFT将频域数据重建为时域
        ifft_result = torch.fft.irfft(masked_freq, n=seq_len, dim=1)
        # 输出映射
        out = self.reconstruct(ifft_result)
        return out

class ModifiedGraphModule(nn.Module):
    def __init__(self, seg_num_x, seg_num_y, sparse_k=4):
        super().__init__()
        self.sparse_k = sparse_k
        # 稀疏图的邻接矩阵参数
        self.adj_weight = nn.Parameter(torch.randn(seg_num_x, seg_num_x) / (seg_num_x ** 0.5))
        # 输出映射
        self.out_proj = nn.Parameter(torch.randn(seg_num_x, seg_num_y) / (seg_num_x ** 0.5))

    def forward(self, x):
        adj = self.adj_weight
        # 保留每行最大的k个值
        topk_values, indices = torch.topk(adj, k=self.sparse_k, dim=-1)
        mask = torch.zeros_like(adj).scatter_(-1, indices, 1.0)
        adj = adj * mask
        adj = F.softmax(adj, dim=-1)

        # 图卷积操作
        out = torch.matmul(torch.matmul(x, adj), self.out_proj)
        return out

class EnhancedModel(nn.Module):
    def __init__(self, configs):
        super(EnhancedModel, self).__init__()

        # 模型参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.freq_len = configs.freq_len
        assert self.model_type in ['linear', 'mlp', 'graph', 'freq']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        )
        self.conv1d2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 8),
            stride=1,
            padding=self.period_len // 8,
            padding_mode="zeros",
            bias=False
        )

        # 特定模型类型的层
        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )
        elif self.model_type == 'graph':
            self.graph = ModifiedGraphModule(self.seg_num_x, self.seg_num_y)
        elif self.model_type == 'freq':
            self.freq = FrequencyModule(self.seg_num_x, self.seg_num_y, self.freq_len)

    def forward(self, x):
        batch_size = x.shape[0]

        # 标准化和维度变换 b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)

        # 一维卷积聚合
        x_conv = self.conv1d(x.reshape(-1, 1, self.seq_len))
        x = x_conv.reshape(-1, self.enc_in, self.seq_len) + x
        x = self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # 降采样: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # 预测
        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)
        elif self.model_type == 'graph':
            y = self.graph(x)
        elif self.model_type == 'freq':
            y = self.freq(x)

        # 上采样: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # 维度变换和反标准化
        y = y.permute(0, 2, 1) + seq_mean

        return y
