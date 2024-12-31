import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # 分离血压列
    data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure'])
    data.loc[data['BMI Category'] == 'Normal Weight', 'BMI Category'] = 'Normal'
    # 标签编码
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    
    return data, label_encoders

def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def visualize_data(data):
    # 创建保存图像的目录
    os.makedirs('./results/figures', exist_ok=True)
    # 可视化 BMI 分类的分布
    plt.figure(figsize=(10, 6))
    data.loc[data['BMI Category'] == 'Normal Weight', 'BMI Category'] = 'Normal'
    sns.countplot(x='BMI Category', data=data)
    plt.title('Distribution of BMI Categories')
    plt.xlabel('BMI Category')
    plt.ylabel('Count')
    
    # 保存图像
    plt.savefig('./results/figures/bmi_category_distribution.png')
    plt.show()
def save_processed_data(data, output_path):
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
