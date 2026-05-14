import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CNN(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels, 
                 time_points,
                 tsne = False):
        super(EEG_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=(3, 3), stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(3, 6), stride=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8,
                               kernel_size=(3, 3), stride=1,
                               padding=1)
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(3, 6), stride=2)

        self.fc_size = self.get_size(num_channels, time_points)

        self.fc1 = nn.Linear(self.fc_size, 5000)
        self.classifier = nn.Linear(5000, num_classes)
        self.tsne = tsne

    def get_size(self, C, N):

        data = torch.ones((1, 1, C, N))
        x = self.conv1(data)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
          
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features

        tok = x.view(x.size(0), -1)
        tok = self.fc1(tok)
        out = self.classifier(tok)

        #return tok, out
        return out