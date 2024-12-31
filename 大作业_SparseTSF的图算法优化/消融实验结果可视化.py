import matplotlib.pyplot as plt
import numpy as np

# 数据提取
models = [
    "Graph", "Graph with Periodic Broadcasting", 
    "Linear with Periodic Broadcasting", 
    "MLP with Periodic Broadcasting", 
    "Single Linear", "Single MLP"
]
mse = [0.354304224527008, 0.3532668166160583, 0.3598916828263355, 
       0.37319016456604004, 0.36146053671836853, 0.37405720353126526]
mae = [0.38516151905059814, 0.3840063810348511, 0.38554659485816956, 
       0.39408260583877563, 0.3880969285964966, 0.3970913887023926]

# 设置柱状图的宽度
x = np.arange(len(models))
width = 0.25

# 创建图表
plt.figure(figsize=(14, 8))

# 绘制柱状图
plt.bar(x - width, mse, width, label='MSE (Bar)', alpha=0.8)
plt.bar(x, mae, width, label='MAE (Bar)', alpha=0.8)

# 绘制折线图
plt.plot(x, mse, marker='o', color='blue', label='MSE (Line)', linewidth=2)
plt.plot(x, mae, marker='o', color='orange', label='MAE (Line)', linewidth=2)

# 添加文本标注
for i in range(len(models)):
    plt.text(x[i] - width, mse[i] + 0.005, f'{mse[i]:.3f}', ha='center', fontsize=10)
    plt.text(x[i], mae[i] + 0.005, f'{mae[i]:.3f}', ha='center', fontsize=10)

# 设置y轴范围以放大差异
plt.ylim(0.34, 0.40)  # 你可以根据你的数据调整此范围

# 添加标题和标签
plt.title("Ablation Experiment Results", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("Performance Metrics", fontsize=14)
plt.xticks(x, models, fontsize=12, rotation=15)
plt.legend(fontsize=12)

# 添加网格以增强可读性
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
plt.show()



