import re
import matplotlib.pyplot as plt

# 初始化列表以存储数据
epochs = []
train_losses = []
val_losses = []

# 定义正则表达式模式
epoch_pattern = re.compile(r'第(\d+)/\d+轮')
# 修改后的 loss_pattern，允许逗号后有可选的空白字符
loss_pattern = re.compile(r'训练集损失：([0-9.]+),\s*验证集损失：([0-9.]+)')

data_path = 'result_streaming.txt'  # 请确保文件路径正确

# 读取文件
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 尝试匹配 epoch 行
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
            continue  # 继续读取下一行

        # 尝试匹配损失行
        loss_match = loss_pattern.search(line)
        if loss_match and epochs:
            train_loss = float(loss_match.group(1))
            val_loss = float(loss_match.group(2))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 调试打印
            print(f"Epoch {epoch}: Train Loss={train_loss}, Val Loss={val_loss}")

# 检查数据是否成功提取
if not epochs:
    print("未能提取到任何 epoch 数据。请检查 result_streaming.txt 文件格式。")
elif not (train_losses and val_losses ):
    print("未能提取到完整的损失数据（训练集、验证集）。请检查 result_streaming.txt 文件格式。")
elif len(epochs) != len(train_losses) or len(epochs) != len(val_losses) :
    print("提取的数据长度不一致，请检查 result_streaming.txt 文件格式。")
else:
    print(f"提取到 {len(epochs)} 个 epoch 的数据。")
    
    # 可选：打印部分数据以确认
    print(f"Epochs: {epochs[:5]} ...")
    print(f"Train Losses: {train_losses[:5]} ...")
    print(f"Validation Losses: {val_losses[:5]} ...")

    # 绘制损失曲线
    plt.figure(figsize=(20, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, max(epochs)+1))  # 设置 x 轴刻度为每个 epoch
    plt.tight_layout()
    plt.savefig('loss_curve_with_test.png')  # 保存为图片
    plt.show()
