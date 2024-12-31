import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 假设模型结构与之前定义的相同
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        y = self.linear(x)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_mean
        return y

# 配置模型的参数
class Config:
    def __init__(self):
        self.seq_len = 720  # 序列长度
        self.pred_len = 720  # 预测长度
        self.enc_in = 7  # 输入通道数
        self.period_len = 24  # 周期长度（例如一天24小时）

# 加载模型
checkpoint_path = "checkpoint.pth"  # 模型检查点路径
checkpoint = torch.load(checkpoint_path)

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
one_hot_input[0, torch.arange(seq_len), 0] = 1  # 生成一个对角线为1的独热编码输入

# 运行模型并获取输出
model_output = model(one_hot_input)

# 提取模型线性层的权重矩阵
weights = model.linear.weight.data.numpy()  # 获取稀疏模型的权重矩阵
print("权重矩阵的形状:", weights.shape)  # 打印权重矩阵的形状

# 可视化权重矩阵（热力图）
plt.figure(figsize=(8, 8))
# 使用'YlOrRd'颜色映射，颜色越深表示值越大，底色为白色
plt.imshow(weights, cmap='YlOrRd', interpolation='nearest', vmin=np.min(weights), vmax=np.max(weights))
plt.colorbar(label='Weight Magnitude')  # 添加颜色条
plt.title("Heatmap of Model Weights")
plt.xlabel("Forecast Horizon (Y-axis)")
plt.ylabel("Look-back Length (X-axis)")
plt.show()

# 保存热力图为图片
plt.savefig("model_weights_heatmap.png")









