import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def evaluate_model(model, X_test_tensor, y_test_tensor,label_encoders):
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_logits, 1)
        accuracy = (y_pred == y_test_tensor).sum().item() / y_test_tensor.size(0)

    # 可视化预测结果与实际结果
    le = label_encoders
    y_test_original = le.inverse_transform(y_test_tensor.numpy())
    y_pred_original = le.inverse_transform(y_pred.numpy().astype(int))
    plt.figure(figsize=(10, 6))
    # 绘制实际值的条形图
    sns.histplot(y_test_original, color='red', label='Actual', bins=3, alpha=0.6, kde=True)

# 绘制预测值的条形图
    sns.histplot(y_pred_original, color='blue', label='Predicted', bins=3, alpha=0.3, kde=True)

    plt.xlabel('BMI Category')
    plt.ylabel('Frequency')
    plt.title('Predicted vs Actual BMI Categories')
    plt.legend()
    plt.savefig('./results/figures/predicted_vs_actual.png')
    plt.show()

    return accuracy
