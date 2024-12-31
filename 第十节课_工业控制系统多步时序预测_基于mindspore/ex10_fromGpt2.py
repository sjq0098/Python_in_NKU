# %% [markdown]
# # 工业时序预测模型（包含偏差残差块）及绘图

# %%
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore.common.initializer import XavierUniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 设置MindSpore运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 设置随机种子以保证结果可重复
mindspore.set_seed(42)
np.random.seed(42)

# 定义传感器数量和预测步数
sensor_num = 23  # 传感器数量
horizon = 5      # 预测的时间步数

# 数据加载与预处理
def load_data(file_path, window_size=30, steps=horizon):
    data = pd.read_csv(file_path)
    
    # 检查数据是否有缺失值
    if data.isnull().sum().sum() > 0:
        print("数据中存在缺失值，进行填充...")
        data = data.fillna(method='ffill').fillna(method='bfill')  # 前向填充和后向填充

    data_values = data.values.astype(np.float32)  # 转换为float32

    X = []
    y = []
    total_steps = window_size + steps
    for i in range(len(data_values) - total_steps + 1):
        X.append(data_values[i:i + window_size, :])       # 输入特征（包含所有传感器数据）
        y.append(data_values[i + window_size:i + total_steps, :])  # 多步标签（所有传感器数据）

    X = np.array(X)  # 形状: (样本数, window_size, 特征数)
    y = np.array(y)  # 形状: (样本数, steps, 特征数)

    return X, y

# 数据路径
data_path = "D:/python_学习文件/第十节课/1731591426495-jl_data_train.csv"

# 参数设置
window_size = 30   # 输入序列的窗口大小
steps = horizon    # 预测的步数
batch_size = 32
num_epochs = 10    # 可以根据需要调整
learning_rate = 1e-3
random_state = 42

# 加载数据
X_data, y_data = load_data(data_path, window_size=window_size, steps=steps)

# 打印数据形状以验证
print(f"Loaded features shape: {X_data.shape}, dtype: {X_data.dtype}")  # (样本数, window_size, 特征数)
print(f"Loaded labels shape: {y_data.shape}, dtype: {y_data.dtype}")    # (样本数, steps, 特征数)

# 数据集划分：训练集、验证集、测试集
def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    total_samples = X.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_data, y_data)

# 打印各个子集的形状
print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# 数据标准化
num_features = X_train.shape[-1]

# 初始化标准化器
feature_scaler = StandardScaler()

# 拟合标准化器仅在训练集上
X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
X_val_scaled = feature_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
X_test_scaled = feature_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

# 对标签进行标准化
y_train_scaled = feature_scaler.transform(y_train.reshape(-1, num_features)).reshape(y_train.shape)
y_val_scaled = feature_scaler.transform(y_val.reshape(-1, num_features)).reshape(y_val.shape)
y_test_scaled = feature_scaler.transform(y_test.reshape(-1, num_features)).reshape(y_test.shape)

# 创建数据集
def create_dataset(features, labels, batch_size=32, shuffle=False):
    dataset = ds.NumpySlicesDataset({'features': features, 'labels': labels}, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = create_dataset(X_train_scaled, y_train_scaled, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(X_val_scaled, y_val_scaled, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(X_test_scaled, y_test_scaled, batch_size=batch_size, shuffle=False)

# 定义空间MLP模块
class SpatialMLP(nn.Cell):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SpatialMLP, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(input_size, hidden_size, weight_init=XavierUniform()))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Dense(input_size, output_size, weight_init=XavierUniform()))
        self.mlp = nn.SequentialCell(layers)

    def construct(self, x):
        return self.mlp(x)

# 定义时间卷积模块
class TemporalConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='valid'):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            weight_init=XavierUniform()
        )

    def construct(self, x):
        return self.conv(x)

# 定义带偏差残差块的多步预测模型
class TCN_MLP_with_Bias_Block(nn.Cell):
    def __init__(self):
        super(TCN_MLP_with_Bias_Block, self).__init__()

        # 空间MLP
        self.spatial_mlp = SpatialMLP(input_size=sensor_num, hidden_sizes=[128, 64, 32], output_size=sensor_num)

        # 时间卷积
        self.tcn = nn.SequentialCell(
            TemporalConv(in_channels=1, out_channels=1, kernel_size=(3, 1)),
            nn.ReLU(),
            TemporalConv(in_channels=1, out_channels=1, kernel_size=(3, 1)),
            nn.ReLU(),
        )

        # 最后的卷积层
        self.final_conv = TemporalConv(in_channels=1, out_channels=1, kernel_size=(26, 1))

        # 偏差块
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64, weight_init=XavierUniform()),
            nn.ReLU(),
            nn.Dense(64, 32, weight_init=XavierUniform()),
            nn.ReLU(),
            nn.Dense(32, sensor_num, weight_init=XavierUniform()),
            nn.ReLU(),
        )
        self.bias_conv = TemporalConv(in_channels=1, out_channels=1, kernel_size=(32, 1))

    def construct(self, x):
        # x: [batch_size, window_size, sensor_num]
        h = self.spatial_mlp(x)  # [batch_size, window_size, sensor_num]
        h = x + h  # 残差连接
        h = h.expand_dims(1)  # [batch_size, 1, window_size, sensor_num]
        h = self.tcn(h)  # [batch_size, 1, new_time_steps, sensor_num]
        y = self.final_conv(h)  # [batch_size, 1, 1, sensor_num]
        y = y.squeeze(2)  # [batch_size, 1, sensor_num]

        # 构建偏差块的输入
        bias_input = ops.Concat(1)((x, y.squeeze(1)))  # [batch_size, window_size + 1, sensor_num]
        bias_output = self.bias_block(bias_input)  # [batch_size, sensor_num]
        bias_output = bias_output.expand_dims(1).expand_dims(2)  # [batch_size, 1, 1, sensor_num]
        bias_output = self.bias_conv(bias_output)  # [batch_size, 1, 1, sensor_num]
        bias_output = bias_output.squeeze(2)  # [batch_size, 1, sensor_num]

        # 校正预测输出
        y = y + bias_output  # [batch_size, 1, sensor_num]

        return y

# 定义损失函数和评估指标
def evaluate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - mse / np.var(y_true)
    return {'MSE': mse, 'MAE': mae, 'R2': r2}

# 定义训练和测试的模型运行类
class MODEL_RUN:
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = grad_fn
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

    def _train_one_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def _train_one_epoch(self, train_dataset):
        self.model.set_train(True)
        epoch_loss = 0.0
        for batch in train_dataset.create_dict_iterator():
            data = batch['features']
            label = batch['labels'][:, -1, :]  # 取最后一个时间步的数据作为标签
            loss = self._train_one_step(data, label)
            epoch_loss += loss.asnumpy()
        avg_loss = epoch_loss / train_dataset.get_dataset_size()
        return avg_loss

    def evaluate(self, dataset):
        self.model.set_train(False)
        ls_pred, ls_label = [], []
        epoch_loss = 0.0
        for batch in dataset.create_dict_iterator():
            data = batch['features']
            label = batch['labels'][:, -1, :]
            pred = self.model(data)
            loss = self.loss_fn(pred.squeeze(1), label)
            epoch_loss += loss.asnumpy()
            ls_pred.append(pred.asnumpy())
            ls_label.append(label.asnumpy())
        avg_loss = epoch_loss / dataset.get_dataset_size()
        preds = np.concatenate(ls_pred, axis=0)
        labels = np.concatenate(ls_label, axis=0)
        return avg_loss, preds, labels

    def train(self, train_dataset, val_dataset, max_epoch_num, ckpt_file_path):
        min_loss = np.inf
        print('开始训练......')
        for epoch in range(1, max_epoch_num + 1):
            train_loss = self._train_one_epoch(train_dataset)
            val_loss, val_preds, val_labels = self.evaluate(val_dataset)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            val_preds_inversed = feature_scaler.inverse_transform(val_preds.squeeze(1))
            val_labels_inversed = feature_scaler.inverse_transform(val_labels)
            metrics = evaluate_metrics(val_labels_inversed, val_preds_inversed)
            self.val_metrics.append(metrics)
            print(f"第{epoch}/{max_epoch_num}轮，训练损失：{train_loss:.4f}，验证损失：{val_loss:.4f}，验证指标：{metrics}")
            if val_loss < min_loss:
                mindspore.save_checkpoint(self.model, ckpt_file_path)
                min_loss = val_loss
        print('训练完成！')

    def test(self, test_dataset, ckpt_file_path):
        mindspore.load_checkpoint(ckpt_file_path, net=self.model)
        test_loss, preds, labels = self.evaluate(test_dataset)
        return test_loss, preds, labels

# 实例化模型
model = TCN_MLP_with_Bias_Block()

# 定义损失函数和优化器
loss_fn = nn.MAELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

# 定义前向函数和梯度函数
def forward_fn(data, label):
    pred = model(data)
    loss = loss_fn(pred.squeeze(1), label)
    return loss, pred

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 创建模型运行对象
model_run = MODEL_RUN(model, loss_fn, optimizer, grad_fn)

# 训练模型
model_run.train(train_dataset=train_dataset, val_dataset=val_dataset, max_epoch_num=num_epochs, ckpt_file_path='tcn_mlp_bias.ckpt')

# 测试模型
test_loss, preds, labels = model_run.test(test_dataset=test_dataset, ckpt_file_path='tcn_mlp_bias.ckpt')
print('测试集损失：{0}'.format(test_loss))

# 逆变换预测值和真实值
preds_inversed = feature_scaler.inverse_transform(preds.squeeze(1))
labels_inversed = feature_scaler.inverse_transform(labels)

# 评估测试集指标
test_metrics = evaluate_metrics(labels_inversed, preds_inversed)
print(f"测试集指标：{test_metrics}")

# 绘制预测结果与真实值对比图（前100个样本）
plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), preds_inversed[:100, 0], label='预测值', color='Red')
plt.plot(range(1, 101), labels_inversed[:100, 0], label='真实值', color='Blue')
plt.xlabel('样本编号')
plt.ylabel('传感器值')
plt.title('预测值与真实值对比（前100个样本）')
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练集和验证集的损失曲线
plt.figure(figsize=(12, 8))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, model_run.train_losses, label='Training Loss')
plt.plot(epochs_range, model_run.val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 绘制验证集的评估指标曲线
mse_vals = [metric['MSE'] for metric in model_run.val_metrics]
mae_vals = [metric['MAE'] for metric in model_run.val_metrics]
r2_vals = [metric['R2'] for metric in model_run.val_metrics]

plt.figure(figsize=(12, 8))
plt.plot(epochs_range, mse_vals, label='Validation MSE')
plt.plot(epochs_range, mae_vals, label='Validation MAE')
plt.plot(epochs_range, r2_vals, label='Validation R2')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 绘制测试集的评估指标
plt.figure(figsize=(6, 4))
metrics_names = list(test_metrics.keys())
metrics_values = list(test_metrics.values())
plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Test Metrics')
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()
