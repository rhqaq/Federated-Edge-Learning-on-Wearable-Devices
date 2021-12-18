import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)        # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=9,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.attn = SelfAttentionEncoder(100)
        self.fc = nn.Linear(100, 15)
    def forward(self, inputs):
        # (N,L,D) batch,时序长度,特征数量
        tensor = self.LSTM(inputs)[0] #(N,L,D)
        tensor = self.attn(tensor)  # (N,D)
        tensor = self.fc(tensor)
        return tensor

class Simple_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=9,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(100, 10)
    def forward(self, inputs):
        # (N,L,D) batch,时序长度,特征数量
        output,(hn,cn) = self.LSTM(inputs) #(N,L,D)
        tensor = self.fc(hn)
        return tensor


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 1000)
        self.fc2 = nn.Linear(1000, 15)

    def forward(self, inputs):
        tensor = inputs.view(-1, 9)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

