import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 256)  # 假设有三个类别
        self.fc4 = nn.Linear(256,48)
        self.fc5 = nn.Linear(48,96)
        self.fc6 = nn.Linear(96,3)
        self.dropout = nn.Dropout(0.3)  # Dropout 概率为 0.5
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x=self.dropout(x)
        x = torch.relu(self.fc2(x))
        #x=self.dropout(x)
        x = torch.relu(self.fc3(x))
        #x=self.dropout(x)
        x = torch.relu(self.fc4(x))
        x=self.dropout(x)
        x = torch.relu(self.fc5(x))
        x=self.dropout(x)
        #print(x.shape)
        x = self.fc6(x)
        return x
