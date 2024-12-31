# src/train.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from src.data_preprocessing import load_data, split_data
from src.model import create_pipelines

def train_and_evaluate(filepath):
    # 加载数据
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 创建模型管道
    pipelines = create_pipelines()
    results = {}

    # 训练和评估模型
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"{name} - MSE: {mse}, MAE: {mae}, R2: {r2}")

    # 选择性能最佳的模型
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = pipelines[best_model_name]
    print("最佳模型为：", best_model_name, "具有R2分数：", results[best_model_name]['R2'])

    # 保存结果
    pd.DataFrame(results).to_csv('D:\python_学习文件\第八节课\ex8\/results/metrics.csv', index=False)
    return best_model
