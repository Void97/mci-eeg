import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes, 
                 num_channels, 
                 time_points,
                 tsne = False):
        super(DeepConvNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=(1, 1), bias=False),
            nn.Conv2d(25, 25, kernel_size=(num_channels, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), stride=(1, 1), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(0.5)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), stride=(1, 1), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(0.5)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), stride=(1, 1), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(0.5)
        )

        # Calculate final feature size dynamically
        self.feature_size = self.get_size(num_channels, time_points)

        self.classifier = nn.Linear(self.feature_size, num_classes)
        
        self.tsne = tsne

    def get_size(self, C, T):
        x = torch.ones((1, 1, C, T))  # shape: [batch, 1, channels, time]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # x: [batch_size, C, T]
        #x = x.unsqueeze(1)  # Add channel dimension → [B, 1, C, T]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features
        
        tok = x.view(x.size(0), -1)
        out = self.classifier(tok)
        #return tok, out
        return out
