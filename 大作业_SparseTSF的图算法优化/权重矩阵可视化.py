import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GraphModule(nn.Module):
    def __init__(self, seg_num_x, seg_num_y, sparse_k=4):
        super().__init__()
        self.sparse_k = sparse_k
        # 稀疏图的邻接矩阵参数
        self.adj_weight = nn.Parameter(torch.randn(seg_num_x, seg_num_x) / (seg_num_x ** 0.5))
        # 输出映射
        self.out_proj = nn.Parameter(torch.randn(seg_num_x, seg_num_y) / (seg_num_x ** 0.5))
        
    def forward(self, x):
        # 构建稀疏邻接矩阵
        adj = self.adj_weight
        # 保留每行最大的k个值
        topk_values, indices = torch.topk(adj, k=self.sparse_k, dim=-1)
        mask = torch.zeros_like(adj).scatter_(-1, indices, 1.0)
        adj = adj * mask
        adj = F.softmax(adj, dim=-1)
        
        # 图卷积操作
        out = torch.matmul(torch.matmul(x, adj), self.out_proj)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp', 'graph']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 1D卷积层
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
            # 修正周期性特征增强的维度
            self.period_enhance = nn.Parameter(
                torch.randn(self.period_len, 1) / self.period_len
            )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 标准化和维度变换 b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D卷积聚合
        x_conv = self.conv1d(x.reshape(-1, 1, self.seq_len))
        
        if self.model_type == 'graph':
            # 确保维度匹配的周期性特征增强
            x_period = x_conv.reshape(-1, self.seq_len // self.period_len, self.period_len)
            period_weights = F.softmax(self.period_enhance, dim=0)  # [period_len, 1]
            x_period = x_period * period_weights.t()  # 广播乘法
            x_conv = x_period.reshape(-1, 1, self.seq_len)
        
        x = x_conv.reshape(-1, self.enc_in, self.seq_len) + x
        x = self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len)+ x
        # 降采样: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # 稀疏预测
        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)
        else:  # graph
            y = self.graph(x)

        # 上采样: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # 维度变换和反标准化
        y = y.permute(0, 2, 1) + seq_mean

        return y

# 配置模型的参数
class Config:
    def __init__(self):
        self.seq_len = 720
        self.pred_len = 720
        self.enc_in = 7
        self.period_len = 24
        self.d_model = 64
        self.model_type = 'graph'  # 选择与checkpoint匹配的类型



# 加载模型
checkpoint_path = "./大作业_SparseTSF的图算法优化/checkpoint.pth"  # 模型检查点路径
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 创建模型实例
model = Model(Config())  # 创建模型实例
model.load_state_dict(checkpoint)  # 如果没有'model_state_dict'，直接加载权重字典
model.eval()  # 设置为评估模式

# 创建独热编码输入
batch_size = 256
seq_len = 720
enc_in = 7

# 创建独热编码矩阵（长度为720的独热向量）
one_hot_input = torch.zeros(batch_size, seq_len, enc_in)
one_hot_input[0, torch.arange(seq_len), 0] = 1  # 第0个通道为独热编码

# 运行模型并获取输出
model_output = model(one_hot_input)

# 根据模型类型可视化权重
if model.model_type == 'linear':
    # 提取模型线性层的权重矩阵
    weights = model.linear.weight.data.numpy()  # [seg_num_x, seg_num_y]
    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='YlOrRd', interpolation='nearest', 
               vmin=np.min(weights), vmax=np.max(weights))
    plt.colorbar(label='Weight Magnitude')
    plt.title("Heatmap of Linear Model Weights")
    plt.xlabel("Forecast Segments")
    plt.ylabel("Look-back Segments")
    plt.savefig("model_weights_heatmap_linear.png")
    plt.show()

elif model.model_type == 'graph':
    # 如果是graph模型, 我们可以可视化adj_weight或out_proj
    # 例如可视化adj_weight
    adj_weights = model.graph.adj_weight.data.numpy()  # [seg_num_x, seg_num_x]
    plt.figure(figsize=(8, 8))
    plt.imshow(adj_weights, cmap='YlOrRd', interpolation='nearest', 
               vmin=np.min(adj_weights), vmax=np.max(adj_weights))
    plt.colorbar(label='Weight Magnitude')
    plt.title("Heatmap of Graph Adjacency Weights")
    plt.xlabel("Look-back Segments")
    plt.ylabel("Look-back Segments")
    plt.savefig("model_weights_heatmap_graph.png")
    plt.show()

elif model.model_type == 'mlp':
    # 对于MLP, 我们可以查看mlp第一层的权重
    mlp_weights = model.mlp[0].weight.data.numpy()  # [d_model, seg_num_x]
    plt.figure(figsize=(8, 8))
    plt.imshow(mlp_weights, cmap='YlOrRd', interpolation='nearest', 
               vmin=np.min(mlp_weights), vmax=np.max(mlp_weights))
    plt.colorbar(label='Weight Magnitude')
    plt.title("Heatmap of MLP First-Layer Weights")
    plt.xlabel("Look-back Segments")
    plt.ylabel("Hidden Dimension")
    plt.savefig("model_weights_heatmap_mlp.png")
    plt.show()










