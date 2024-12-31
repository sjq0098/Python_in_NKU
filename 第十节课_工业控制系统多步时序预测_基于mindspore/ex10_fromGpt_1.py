import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置随机种子以保证结果可重复
mindspore.set_seed(42)

# 数据加载与预处理
def load_data(file_path, window_size=10, steps=3):
    """
    加载CSV数据，并生成适合多步预测的输入和标签。

    :param file_path: CSV文件路径
    :param window_size: 输入序列的窗口大小（过去的时间步数）
    :param steps: 预测的步数（未来的时间步数）
    :return: 输入特征和多步标签
    """
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
        X.append(data_values[i:i + window_size, :-1])  # 输入特征（不包括标签）
        y.append(data_values[i + window_size:i + total_steps, -1])  # 多步标签（仅标签列）

    X = np.array(X)  # 形状: (样本数, window_size, 特征数)
    y = np.array(y)  # 形状: (样本数, steps)

    # 为了适应模型输入格式，增加一个维度
    X = np.expand_dims(X, axis=1)  # 形状: (样本数, 1, window_size, 特征数)

    return X, y

# 数据路径
data_path = "D:/python_学习文件/第十节课/1731591426495-jl_data_train.csv"

# 参数设置
window_size = 10  # 输入序列的窗口大小
steps = 3         # 预测的步数
batch_size = 32
num_epochs = 50
learning_rate = 0.001
test_size = 0.2
val_size = 0.1
random_state = 42

# 加载数据
X_data, y_data = load_data(data_path, window_size=window_size, steps=steps)

# 打印数据形状以验证
print(f"Loaded features shape: {X_data.shape}, dtype: {X_data.dtype}")  # (样本数, 1, window_size, 特征数)
print(f"Loaded labels shape: {y_data.shape}, dtype: {y_data.dtype}")    # (样本数, steps)

# 数据集划分：训练集、验证集、测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, test_size=test_size, random_state=random_state
)
val_relative_size = val_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state
)

# 打印各个子集的形状
print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# 数据标准化
# 由于 X 是 4D 的 (samples, 1, window_size, features)，我们需要先将其转换为 2D
num_features = X_train.shape[-1]

X_train_reshaped = X_train.reshape(-1, num_features)  # (samples*1*window_size, features)
X_val_reshaped = X_val.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

# 初始化标准化器
feature_scaler = StandardScaler()
label_scaler = StandardScaler()

# 拟合标准化器仅在训练集上
X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)

# 对标签进行标准化
y_train_scaled = label_scaler.fit_transform(y_train)
y_val_scaled = label_scaler.transform(y_val)
y_test_scaled = label_scaler.transform(y_test)

# 创建数据集
def create_dataset(features, labels, batch_size=32, shuffle=False):
    """
    创建MindSpore数据集。

    :param features: 输入特征
    :param labels: 多步标签
    :param batch_size: 批大小
    :param shuffle: 是否打乱数据
    :return: MindSpore数据集
    """
    dataset = ds.NumpySlicesDataset({'features': features, 'labels': labels}, shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = create_dataset(X_train_scaled, y_train_scaled, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(X_val_scaled, y_val_scaled, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(X_test_scaled, y_test_scaled, batch_size=batch_size, shuffle=False)

# 定义空间MLP模块
class SpatialMLP(nn.Cell):
    def __init__(self, input_size, hidden_sizes):
        super(SpatialMLP, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Dense(input_size, input_size))  # 输出维度保持不变
        self.mlp = nn.SequentialCell(layers)

    def construct(self, x):
        return self.mlp(x)

# 定义时间卷积模块，修正kernel_size为(1, k)以匹配输入形状
class TemporalConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(TemporalConv, self).__init__()
        # 修改kernel_size为(1, kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            pad_mode=padding
        )

    def construct(self, x):
        return self.conv(x)

# 定义多步预测模型
class MultiStepAwareTCN_MLP(nn.Cell):
    def __init__(self, input_feature_size, window_size, hidden_sizes_mlp=[128, 64, 32], hidden_sizes_residual=[64, 32], steps=3):
        super(MultiStepAwareTCN_MLP, self).__init__()
        self.steps = steps
        self.window_size = window_size

        # 空间MLP
        self.spatial_mlp = SpatialMLP(input_size=input_feature_size * window_size, hidden_sizes=hidden_sizes_mlp)

        # 时间卷积
        self.temporal_conv1 = TemporalConv(in_channels=1, out_channels=1, kernel_size=3)
        self.temporal_conv2 = TemporalConv(in_channels=1, out_channels=1, kernel_size=26)

        # 残差块，修正input_size为hidden_sizes_mlp[-1]（即32）
        self.residual_spatial_mlp = SpatialMLP(input_size=hidden_sizes_mlp[-1], hidden_sizes=hidden_sizes_residual)
        self.temporal_conv_residual = TemporalConv(in_channels=1, out_channels=1, kernel_size=32)

        # 输出层，修正input_size为hidden_sizes_residual[-1]（即32）
        self.output_layer = nn.Dense(hidden_sizes_residual[-1], steps)  # 输出多步

    def construct(self, x):
        batch_size = x.shape[0]
        # x: [batch_size, 1, window_size, feature_size]
        x = x.reshape(batch_size, -1)  # [batch_size, window_size * feature_size]
        x = self.spatial_mlp(x)  # [batch_size, hidden_size=32]

        # 调整维度以匹配时间卷积输入要求
        x = ops.expand_dims(x, 1)  # [batch_size, 1, 32]
        x = ops.expand_dims(x, 2)  # [batch_size, 1, 1, 32]

        # 残差连接
        res = x  # [batch_size, 1, 1, 32]
        x = self.temporal_conv1(x)  # [batch_size, 1, 1, 32]
        x = ops.add(x, res)  # [batch_size, 1, 1, 32]

        # 时间卷积
        x = self.temporal_conv2(x)  # [batch_size, 1, 1, 32]

        # 残差块
        res = x  # [batch_size, 1, 1, 32]
        x = x.reshape(batch_size, -1)  # [batch_size, 32]
        x = self.residual_spatial_mlp(x)  # [batch_size, 32]
        x = ops.expand_dims(x, 1)  # [batch_size, 1, 32]
        x = ops.expand_dims(x, 2)  # [batch_size, 1, 1, 32]
        x = self.temporal_conv_residual(x)  # [batch_size, 1, 1, 32]
        x = ops.add(x, res)  # [batch_size, 1, 1, 32]

        # 输出层
        x = x.reshape(batch_size, -1)  # [batch_size, 32]
        x = self.output_layer(x)  # [batch_size, steps=3]
        return x

# 定义评估指标函数
def evaluate_metrics(y_true, y_pred):
    """
    计算回归评估指标：MSE, MAE, R2

    :param y_true: 真实值，形状 [num_samples, steps]
    :param y_pred: 预测值，形状 [num_samples, steps]
    :return: dict 包含各指标
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R2': r2}

# 模型实例化与训练准备
input_feature_size = X_train.shape[-1]  # 特征数
model = MultiStepAwareTCN_MLP(
    input_feature_size=input_feature_size,
    window_size=window_size,
    hidden_sizes_mlp=[128, 64, 32],
    hidden_sizes_residual=[64, 32],
    steps=steps
)
loss_fn = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

# 将模型和损失函数结合起来
loss_net = nn.WithLossCell(model, loss_fn)

# 使用 TrainOneStepCell 包装 loss_net 和 optimizer
train_network = nn.TrainOneStepCell(loss_net, optimizer)
train_network.set_train()

# 训练循环
train_losses = []
val_losses = []
val_metrics = []
test_metrics = []

for epoch in range(1, num_epochs + 1):
    epoch_train_loss = 0.0
    # 训练阶段
    for batch in train_dataset.create_dict_iterator():
        features = batch['features']  # [batch_size, 1, window_size, feature_size]
        labels = batch['labels']      # [batch_size, steps]

        # 训练一步
        loss = train_network(features, labels)
        epoch_train_loss += loss.asnumpy()

    avg_train_loss = epoch_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.set_train(False)  # 切换到评估模式
    y_val_true = []
    y_val_pred = []
    epoch_val_loss = 0.0
    for batch in val_dataset.create_dict_iterator():
        features = batch['features']
        labels = batch['labels']
        predictions = model(features)
        loss = loss_fn(predictions, labels)
        epoch_val_loss += loss.asnumpy()

        y_val_true.append(labels.asnumpy())
        y_val_pred.append(predictions.asnumpy())

    avg_val_loss = epoch_val_loss / len(val_dataset)
    val_losses.append(avg_val_loss)

    # 计算验证集的评估指标
    y_val_true = np.concatenate(y_val_true, axis=0)
    y_val_pred = np.concatenate(y_val_pred, axis=0)
    y_val_pred_inversed = label_scaler.inverse_transform(y_val_pred)  # 逆变换预测值
    y_val_true_inversed = label_scaler.inverse_transform(y_val_true)  # 逆变换真实值
    metrics = evaluate_metrics(y_val_true_inversed, y_val_pred_inversed)
    val_metrics.append(metrics)

    model.set_train(True)  # 切换回训练模式

    print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f},Test Loss: { avg_val_loss:.4f}, Validation Loss: {avg_val_loss:.4f},  Validation Metrics: {metrics}")

print("训练完成！")

# 测试阶段
model.set_train(False)
y_test_true = []
y_test_pred = []
for batch in test_dataset.create_dict_iterator():
    features = batch['features']
    labels = batch['labels']
    predictions = model(features)
    y_test_true.append(labels.asnumpy())
    y_test_pred.append(predictions.asnumpy())

y_test_true = np.concatenate(y_test_true, axis=0)
y_test_pred = np.concatenate(y_test_pred, axis=0)
y_test_pred_inversed = label_scaler.inverse_transform(y_test_pred)  # 逆变换预测值
y_test_true_inversed = label_scaler.inverse_transform(y_test_true)  # 逆变换真实值
test_metrics_dict = evaluate_metrics(y_test_true_inversed, y_test_pred_inversed)
test_metrics.append(test_metrics_dict)
print(f"Test Metrics: {test_metrics_dict}")

# 绘制训练集和验证集的损失曲线
plt.figure(figsize=(12, 8))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 绘制验证集的评估指标曲线
mse_vals = [metric['MSE'] for metric in val_metrics]
mae_vals = [metric['MAE'] for metric in val_metrics]
r2_vals = [metric['R2'] for metric in val_metrics]

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
metrics_names = list(test_metrics_dict.keys())
metrics_values = list(test_metrics_dict.values())
plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Test Metrics')
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()








