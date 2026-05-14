import torch
import math
import torch.nn as nn

class SCCNet(nn.Module):
    def __init__(self,
                 num_classes, 
                 C, 
                 N, 
                 sfreq,  
                 Nu=None, 
                 Nt=1, 
                 Nc=20,
                 tsne = False
                 ):
        super(SCCNet, self).__init__()

        Nu = C if Nu is None else Nu
        self.fs1 = int(math.floor(sfreq * 0.1))

        self.conv1 = nn.Conv2d(1, Nu, (C, Nt))
        self.bn1 = nn.BatchNorm2d(Nu)
        
        self.conv2 = nn.Conv2d(Nu, Nc, (1, int(self.fs1)),
                               padding = (0, int(self.fs1 / 2)))
        self.bn2 = nn.BatchNorm2d(Nc)
        self.drop = nn.Dropout(0.5)
        
        self.avg_pool = nn.AvgPool2d((1, int(sfreq / 2)), stride=(1, int(self.fs1)))
        self.fc_size = self.get_size(C, N)

        self.classifier = nn.Linear(self.fc_size, num_classes, bias = True)

        self.tsne = tsne
        
    def get_size(self, C, N):
        data = torch.ones((1, 1, C, N))
        x = self.conv1(data)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.avg_pool(x)
    
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = x ** 2
        x = self.drop(x)

        x = self.avg_pool(x)
        x = torch.log(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features

        tok = x.view(x.size(0), -1)
        out = self.classifier(tok)

        #return tok, out
        return out
