import numpy as np
import matplotlib.pyplot as plt
import re

# 定义文件路径
file_path = 'result.txt'

# 初始化数据存储
categories = []
mse_values = []
mae_values = []
rse_values = []

# 定义标签格式化函数
def optimize_label(label):
    # 使用正则表达式提取模型名称和其相关信息
    match = re.match(r"^(ETT\w+|Electricity|weather)_\d+_(\d+).*_(linear|mlp).*", label)
    
    if match:
        # 提取模型类型（ETT, Electricity, weather等）
        optimized_label = match.group(1).replace('h', 'm')  # 将 "h" 替换为 "m"，例如 "ETTh1" -> "ETTm1"
        
        # 提取第二个数字（例如720、96等）
        optimized_label += f"_{match.group(2)}"  # 例如 "720"
        
        # 判断模型是linear还是mlp
        if match.group(3) == "linear":
            optimized_label += "_linear"
        elif match.group(3) == "mlp":
            optimized_label += "_mlp"
        
        return optimized_label
    else:
        return label  # 如果没有匹配到，则返回原标签

# 读取文件并提取数据
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # 使用正则表达式匹配模型名称和对应的 mse, mae, rse
        if match := re.match(r'^([\w\d_]+)_.*$', line.strip()):
            # 格式化标签名，调用优化函数
            formatted_category = optimize_label(match.group(1))
            categories.append(formatted_category)
        elif "mse:" in line:
            mse_match = re.search(r'mse:(\d+\.\d+)', line)
            mae_match = re.search(r'mae:(\d+\.\d+)', line)
            rse_match = re.search(r'rse:(\d+\.\d+)', line)
            if mse_match and mae_match and rse_match:
                mse = float(mse_match.group(1))
                mae = float(mae_match.group(1))
                rse = float(rse_match.group(1))
                mse_values.append(mse)
                mae_values.append(mae)
                rse_values.append(rse)

# 检查数据是否为空
if len(categories) == 0 or len(mse_values) == 0 or len(mae_values) == 0 or len(rse_values) == 0:
    raise ValueError("数据不足，无法绘制雷达图。请检查输入文件是否包含正确的数据格式。")

# 将类别数量转换为角度（雷达图使用极坐标）
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# 将数据转换为雷达图格式
mse_values += mse_values[:1]
mae_values += mae_values[:1]
rse_values += rse_values[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(16, 12), subplot_kw=dict(polar=True))

# 绘制MSE、MAE、RSE的雷达图线条和填充
ax.fill(angles, mse_values, color='b', alpha=0.1)
ax.plot(angles, mse_values, color='b', linewidth=2, label='MSE')
ax.fill(angles, mae_values, color='g', alpha=0.1)
ax.plot(angles, mae_values, color='g', linewidth=2, label='MAE')
ax.fill(angles, rse_values, color='r', alpha=0.1)
ax.plot(angles, rse_values, color='r', linewidth=2, label='RSE')

# 设置标签
ax.set_xticks(angles[:-1])

# 调整标签的旋转角度，并增大字体
ax.set_xticklabels(categories, fontsize=12, ha='center', rotation=45)

# 设置Y轴标签
ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontsize=10)

# 添加标题和图例
plt.title('Radar Chart for MSE, MAE, and RSE', fontsize=18)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# 自动调整图形布局，防止标签被遮挡
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 显示图形
plt.tight_layout()
plt.show()






