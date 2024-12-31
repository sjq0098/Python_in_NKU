import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphModule(nn.Module):
    def __init__(self, seg_num_x, seg_num_y, sparse_k=4):
        super().__init__()
        self.sparse_k = sparse_k
        self.adj_weight = nn.Parameter(torch.randn(seg_num_x, seg_num_x) / (seg_num_x ** 0.5))
        self.out_proj = nn.Parameter(torch.randn(seg_num_x, seg_num_y) / (seg_num_x ** 0.5))
        
    def forward(self, x):
        adj = self.adj_weight
        topk_values, indices = torch.topk(adj, k=self.sparse_k, dim=-1)
        mask = torch.zeros_like(adj).scatter_(-1, indices, 1.0)
        adj = adj * mask
        adj = F.softmax(adj, dim=-1)
        out = torch.matmul(torch.matmul(x, adj), self.out_proj)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.period_len = configs['period_len']
        self.d_model = configs['d_model']
        self.model_type = configs['model_type']

        assert self.model_type in ['linear', 'mlp', 'graph']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, 
                                kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, 
                                padding_mode="zeros", bias=False)

        self.conv1d2 = nn.Conv1d(in_channels=1, out_channels=1, 
                                 kernel_size=1 + 2 * (self.period_len // 8),
                                 stride=1, padding=self.period_len // 8, 
                                 padding_mode="zeros", bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )
        elif self.model_type == 'graph':
            self.graph = GraphModule(self.seg_num_x, self.seg_num_y)
            self.period_enhance = nn.Parameter(
                torch.randn(self.period_len, 1) / self.period_len
            )

    def forward(self, x):
        batch_size = x.shape[0]
        
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x_conv = self.conv1d(x.reshape(-1, 1, self.seq_len))

        if self.model_type == 'graph':
            x_period = x_conv.reshape(-1, self.seq_len // self.period_len, self.period_len)
            period_weights = F.softmax(self.period_enhance, dim=0)  
            x_period = x_period * period_weights.t()  
            x_conv = x_period.reshape(-1, 1, self.seq_len)

        x = x_conv.reshape(-1, self.enc_in, self.seq_len) + x
        x = self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)
        else:  # graph
            y = self.graph(x)

        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_mean

        return y

# 定义待测试的Horizon和Look-back列表
horizons = [96, 192, 336, 720]
look_backs = [96, 192, 336, 720]

# 固定其他参数
base_config = {
    'enc_in': 7,
    'period_len': 24,
    'd_model': 512,
    'model_type': 'graph'  # 可根据需要修改
}

# 打印表头
# Horizon \ Look-back 构造表格
header = "Horizon\\Look-back"
print(f"{header:<14}", end="")
for lb in look_backs:
    print(f"{lb:>8}", end="")
print()

for horizon in horizons:
    print(f"{horizon:<14}", end="")
    for lb in look_backs:
        config = base_config.copy()
        config['seq_len'] = lb
        config['pred_len'] = horizon
        model = Model(config)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_params:>8}", end="")
    print()
