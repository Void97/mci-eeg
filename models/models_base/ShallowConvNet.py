import torch
import torch.nn as nn

class ShawllowConvNet(nn.Module):
    def __init__(self,
                 num_classes, 
                 C, 
                 N, 
                 NT = 40, 
                 NS = 40, 
                 kernel_l = 12,
                 pool_l = 35, 
                 pool_t_step = 7,
                 tsne = False):
        super(ShawllowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, NT, (1, kernel_l), bias=False)
        self.conv2 = nn.Conv2d(NT, NS, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(NS)
        self.avg_pool = nn.AvgPool2d((1, pool_l), stride=(1, pool_t_step))
        self.drop = nn.Dropout(0.25)

        self.fc_size = self.get_size(C, N)

        self.classifier = nn.Linear(self.fc_size, num_classes, bias=True)
        #self.batch_norm = batch_norm
        self.tsne = tsne

    def get_size(self, C, N):
        data = torch.ones((1, 1, C, N))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x ** 2
        x = self.avg_pool(x)
        x = torch.log(x)
        x = self.drop(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features

        tok = x.view(x.size(0), -1)
        out = self.classifier(tok)

        #return tok, out
        return out

