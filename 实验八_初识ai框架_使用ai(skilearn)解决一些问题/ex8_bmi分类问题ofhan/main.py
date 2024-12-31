import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.train import train_model
from src.evaluate import evaluate_model
from src.data_preprocessing import load_data, preprocess_data, scale_features, visualize_data, save_processed_data
import matplotlib.pyplot as plt
def main():
    # 加载和预处理数据
    
    data = load_data('./data/data.csv')
    visualize_data(data)
    data, label_encoders = preprocess_data(data)
    # 可视化数据
    

    # 保存处理后的数据
    save_processed_data(data, './data/processed_data.csv')

    features = data.drop(columns=['BMI Category'])
    target = data['BMI Category']

    features_scaled = scale_features(features)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # 转换为PyTorch的Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train.values)
    y_test_tensor = torch.LongTensor(y_test.values)

    # 训练模型
    input_size = X_train.shape[1]
    model, training_losses, training_accuracies = train_model(X_train_tensor, y_train_tensor, input_size)

    # 评估模型
    accuracy = evaluate_model(model, X_test_tensor, y_test_tensor,label_encoders['BMI Category'])
    print(f'Accuracy: {accuracy:.4f}')

    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('./results/figures/training_loss.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.savefig('./results/figures/training_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()