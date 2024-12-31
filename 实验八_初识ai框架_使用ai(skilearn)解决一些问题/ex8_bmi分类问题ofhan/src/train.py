import torch
import torch.nn as nn
import torch.optim as optim
from src.model import MLP
from src.data_preprocessing import load_data, preprocess_data, scale_features

def train_model(X_train_tensor, y_train_tensor, input_size, num_epochs=100):
    model = MLP(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    training_losses = []
    training_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # 记录损失
        training_losses.append(loss.item())

        # 计算训练准确率
        _, y_pred = torch.max(outputs, 1)
        accuracy = (y_pred == y_train_tensor).sum().item() / y_train_tensor.size(0)
        training_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return model, training_losses, training_accuracies
