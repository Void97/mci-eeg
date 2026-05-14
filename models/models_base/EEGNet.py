import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class EEGnet(nn.Module):
    def __init__(self, 
                 num_classes,
                 num_channels,
                 time_points,
                 half_sfreq,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 tsne = False
                 ):
        super(EEGnet, self).__init__()

        self.dropout_rate=dropout_rate
        self.F1 = F1
        self.F2 = F2
        self.D = D
        

        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, half_sfreq), 
                      padding='valid', bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (num_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1,
                        (1, math.floor(half_sfreq / 4)),
                        padding='valid', groups=self.D * self.F1,
                        bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # ← always safe, avoids output size = 1
            nn.Dropout(dropout_rate)
        )

        self.fc_size = self.get_size(num_channels, time_points)
        self.classifier = nn.Linear(self.fc_size, num_classes, bias=True)
        
        self.tsne = tsne
    
    def get_size(self, num_channels, num_time_points):
        dummy_data = torch.ones((1, 1, num_channels, num_time_points))
        x = self.temp_conv(dummy_data)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.temp_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features
        
        tok = x.view(x.size(0), -1)
        out = self.classifier(tok)
        #return tok, out
        return out

    