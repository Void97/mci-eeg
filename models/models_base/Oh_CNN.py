import torch
import math
import torch.nn as nn

class Oh_CNN(nn.Module):
    def __init__(self,
                 num_classes,  
                 num_channels, 
                 time_points,
                 tsne = False):
        super(Oh_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=5,
                               kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5,
                               kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=5, out_channels=5,
                               kernel_size=3, stride=1)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.drop2 = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)  
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)  
        

        self.globalpool = nn.AdaptiveAvgPool1d(1)
        fc_size = self.get_size(num_channels, time_points)

        self.classifier = nn.Linear(in_features=fc_size, out_features=num_classes)
        
        self.leakyReLU = nn.LeakyReLU()
        self.drop3 = nn.Dropout(0.5)
        
        self.tsne = tsne

    def get_size(self, C, N):
        data = torch.ones((1, C, N))
        x = self.conv1(data)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.avgpool1(x)
        x = self.conv4(x)
        x = self.avgpool2(x)
        x = self.conv5(x)
        x = self.globalpool(x)
    
        return x.view(1, -1).shape[1]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyReLU(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.leakyReLU(x)

        x = self.maxpool2(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.leakyReLU(x)

        x = self.avgpool1(x)
        x = self.drop2(x)

        x = self.conv4(x)
        x = self.leakyReLU(x)

        x = self.avgpool2(x)

        x = self.conv5(x)
        x = self.leakyReLU(x)

        x = self.globalpool(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features

        tok = x.view(x.size(0), -1)
        tok = self.drop3(tok)
        out = self.classifier(tok)

        #return tok, out
        return out

