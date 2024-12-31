# main.py

from src.train import train_and_evaluate
from src.evaluate import plot_results
from src.data_preprocessing import load_data, split_data
import pandas as pd
# 定义数据路径
DATA_PATH = 'D:\python_学习文件\第八节课\ex8\data\Sleep_health_and_lifestyle_dataset.csv'

# 训练模型
best_model = train_and_evaluate(DATA_PATH)

# 使用最佳模型进行预测
X, y = load_data(DATA_PATH)
_, X_test, _, y_test = split_data(X, y)
y_pred = best_model.predict(X_test)

# 绘制评估结果
plot_results(y_test, y_pred)
# 输出预测结果
print(y_pred)

#用具体例子预测
new_data = pd.DataFrame({
    'Gender': ['Male'],
    'Age': [25],
    'Occupation': ['Engineer'],
    'Quality of Sleep': [7],
    'Physical Activity Level': [40],
    'Stress Level': [5],
    'BMI Category': ['Normal'],
    'Blood Pressure': ['120/80'],
    'Heart Rate': [70],
    'Daily Steps': [8000],
    'Sleep Disorder': [None]
})
new_prediction = best_model.predict(new_data)
print(f"预测的睡眠时间: {new_prediction[0]}")