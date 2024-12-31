from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 读取数据集
data = pd.read_csv("D:\python_学习文件\第八节课\Sleep_health_and_lifestyle_dataset.csv")
# 选择特征和目标变量
X = data.drop(['Sleep Duration', 'Person ID'], axis=1)
y = data['Sleep Duration']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对分类特征进行独热编码，对数值特征进行标准化
categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
numerical_features = ['Age', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 创建管道，包含预处理和模型
pipelines = {
    'linear_regression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
    'decision_tree': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor())]),
    'random_forest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor())])
}

# 训练模型并评估
results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    print(f"{name} - MSE: {mse}, MAE: {mae}, R2: {r2}")

# 选择性能最佳的模型（这里以R2分数为例，可根据需求调整）
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = pipelines[best_model_name]

# 使用最佳模型进行预测（这里可以是新的数据点）
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
print("最优的模型是："
      f"{best_model_name}, "
      f"MSE: {results[best_model_name]['MSE']}, "
      f"MAE: {results[best_model_name]['MAE']}, "
      f"R2: {results[best_model_name]['R2']}")
# 可视化预测结果与真实值
plt.scatter(y_test, y_pred)
plt.xlabel('ture sleep time')
plt.ylabel('predicted sleep time')
plt.title('regression result')
plt.savefig('./predicted_vs_actual.png')
plt.show()
#绘制预测值与真实值的散点图
print("预测准确率为：", r2_score(y_test, y_pred))