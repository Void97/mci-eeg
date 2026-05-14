import torch
import torch.nn as nn
import torch.nn.functional as F

class SzHNN(nn.Module):
    def __init__(self, num_classes, 
                 num_channels, 
                 time_points=500, 
                 sfreq=500,
                 tsne = False):
        super(SzHNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=5, 
                               kernel_size=15, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10,
                               kernel_size=10, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=10, hidden_size=32,
                            num_layers=1, batch_first=True)
        
        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, num_classes)
        
        self.tsne = tsne

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.permute(0,2,1)

        x,_ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        
        if self.tsne:
            features = x
            return features
        
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        tok = x.view(x.size(0), -1)
        out = F.softmax(tok, dim=1)

        #return tok, out
        return out
