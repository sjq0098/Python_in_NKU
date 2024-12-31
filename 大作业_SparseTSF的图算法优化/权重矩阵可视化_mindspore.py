import mindspore
from mindspore import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

# 配置类，用于模型的参数设置
class Config:
    def __init__(self):
        self.seq_len = 720  # 序列长度
        self.pred_len = 720  # 预测长度
        self.enc_in = 7  # 输入通道数
        self.period_len = 24  # 周期长度（例如一天24小时）
        self.d_model = 128  # MLP模型中的隐藏层大小
        self.model_type = 'linear'  # 或 'mlp'

# 模型定义
class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 获取模型配置参数
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

# 创建配置对象
config = Config()
model = Model(config)

# 加载模型检查点
checkpoint_path = "checkpoint.ckpt"  # 您的checkpoint文件路径
checkpoint = load_checkpoint(checkpoint_path)
load_param_into_net(model, checkpoint)  # 将权重加载到模型中

# 设置模型为评估模式
model.set_train(False)

# 创建独热编码输入
batch_size = 256
seq_len = 720
enc_in = 7

# 创建独热编码矩阵（长度为96的独热向量）
one_hot_input = np.zeros((batch_size, seq_len, enc_in), dtype=np.float32)
one_hot_input[0, np.arange(seq_len), 0] = 1  # 生成一个对角线为1的独热编码输入
one_hot_input = Tensor(one_hot_input)

# 运行模型并获取输出
model_output = model(one_hot_input)

# 根据模型类型提取权重矩阵
if config.model_type == 'linear':
    weights = model.linear.weight.asnumpy()  # 获取线性层的权重矩阵
elif config.model_type == 'mlp':
    weights = model.mlp[0].weight.asnumpy()  # 获取MLP的第一层权重矩阵

print("权重矩阵的形状:", weights.shape)  # 打印权重矩阵的形状

# 可视化权重矩阵（热力图）
plt.figure(figsize=(8, 8))
# 使用'RdYlBu'颜色映射，从红色到白色，深红色表示高值
plt.imshow(weights, cmap='RdYlBu', interpolation='nearest', vmin=np.min(weights), vmax=np.max(weights))
plt.colorbar(label='Weight Magnitude')  # 添加颜色条

# 设置坐标轴范围为 [0, 80] 来确保是正方形
plt.xlim(0, 30)
plt.ylim(0, 30)

# 添加标题和坐标标签
plt.title("Heatmap of Model Weights")
plt.xlabel("Forecast Horizon (Y-axis)")
plt.ylabel("Look-back Length (X-axis)")

# 显示图像
plt.show()

# 保存热力图为图片
plt.savefig("model_weights_heatmap_mindspore.png")

