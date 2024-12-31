import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# 读取日志文件并解析数据
log_file_path = 'output.log'

def parse_log_file(file_path):
    # 初始化数据结构，用于存储日志中提取的数据
    data = {'epoch': [], 'loss': [], 'set': [], 'length': [], 'dataset': []}
    
    with open(file_path, 'r') as file:
        current_dataset = None
        current_length = None
        for line in file:
            # 检查是否有开始新数据集训练的标识，例如 "start training : ETTm1_720_96_SparseTSF_ETTm1"
            dataset_match = re.search(r'start training : ([A-Za-z0-9_]+)_sl(\d+)_pl(\d+)', line)
            if dataset_match:
                current_dataset = dataset_match.group(1)
                current_length = dataset_match.group(3)  # 使用预测长度作为不同组合的区分
                print(f"Detected dataset: {current_dataset} with prediction length: {current_length}")
                continue
            
            # 匹配类似于 "Epoch: 1, Steps: 131 | Train Loss: 0.3599530 Vali Loss: 0.5022938 Test Loss: 0.4047187"
            match = re.search(r'Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d\.]+) Vali Loss: ([\d\.]+) Test Loss: ([\d\.]+)', line)
            if match and current_dataset:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                test_loss = float(match.group(4))

                # 将数据添加到字典中
                data['epoch'].append(epoch)
                data['loss'].append(train_loss)
                data['set'].append('Train')
                data['length'].append(current_length)
                data['dataset'].append(current_dataset)

                data['epoch'].append(epoch)
                data['loss'].append(val_loss)
                data['set'].append('Val')
                data['length'].append(current_length)
                data['dataset'].append(current_dataset)

                data['epoch'].append(epoch)
                data['loss'].append(test_loss)
                data['set'].append('Test')
                data['length'].append(current_length)
                data['dataset'].append(current_dataset)
            else:
                # 如果一行没有匹配成功，打印该行用于调试
                print(f"Line did not match: {line.strip()}")
    
    df = pd.DataFrame(data)
    # 打印数据帧的一部分用于调试
    print(df.head())
    return df

# 将日志文件解析为数据帧
df = parse_log_file(log_file_path)

# 创建用于保存图像的文件夹
output_folder = 'loss_comparison_plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 为每个数据集绘制不同预测长度的组合图像
def plot_loss_comparison_by_dataset(df):
    if df.empty:
        print("The DataFrame is empty. No data available for plotting.")
        return

    plt.style.use('ggplot')
    unique_datasets = df['dataset'].unique()

    for dataset in unique_datasets:
        subset_df = df[df['dataset'] == dataset]
        unique_lengths = subset_df['length'].unique()
        unique_sets = subset_df['set'].unique()

        # 为每个数据集创建一张图
        plt.figure(figsize=(14, 8))

        # 按预测长度将不同的训练/验证/测试数据绘制到同一张图上
        for length in unique_lengths:
            for data_set in unique_sets:
                subset = subset_df[(subset_df['length'] == length) & (subset_df['set'] == data_set)]
                if not subset.empty:
                    # 确保每个epoch对应的点是按顺序排列的
                    subset_sorted = subset.sort_values(by='epoch')
                    plt.plot(subset_sorted['epoch'], subset_sorted['loss'], linestyle='-', marker='o', label=f'{data_set}-{length}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for {dataset} with Different Prediction Lengths')
        plt.legend()

        # 保存图像，使用数据集名称命名
        output_path = os.path.join(output_folder, f'{dataset}_Linear_Loss_Comparison.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")

# 绘制并保存图像
plot_loss_comparison_by_dataset(df)




