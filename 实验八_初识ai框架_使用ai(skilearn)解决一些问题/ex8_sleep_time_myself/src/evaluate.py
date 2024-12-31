# src/evaluate.py

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Sleep Duration')
    plt.ylabel('Predicted Sleep Duration')
    plt.title('Prediction Results')
    plt.savefig('D:\python_学习文件\第八节课\ex8/results/predicted_vs_actual.png')
    plt.show()
    print("预测准确率为：", r2_score(y_test, y_pred))
