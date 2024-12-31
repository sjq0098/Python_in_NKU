# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(['Sleep Duration', 'Person ID'], axis=1)
    y = data['Sleep Duration']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data():
    # 定义数值和分类特征
    categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
    numerical_features = ['Age', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor
