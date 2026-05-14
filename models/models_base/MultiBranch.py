import torch
import math
import torch.nn as nn
import warnings

def conv_out(sample, kernel, stride=1, padding=0):
    return math.floor((sample + 2 * padding - (kernel - 1) - 1) / stride + 1)

def pool_out(sample, kernel, stride=1, padding=0):
    return math.floor((sample + 2 * padding - kernel) / stride + 1)

def padding_same_tuple(kernel):
    kernel -= 1
    if kernel % 2 == 0:
        target = kernel / 2
    else:
        target = (kernel + 1) / 2
    target = int(target)
    if kernel % 2 == 0:
        return (target, target, 0, 0)
    else:
        return (target, target - 1, 0, 0)

class Multi(nn.Sequential):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, eeg_ksize, spatial_channel, tsne, temporal_channel=20, embedded_size=8):
        super().__init__(
            dimChecker(4, 1),
            Stack(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size),
            ClassificationHead(n_classes, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size, tsne)
        )

    def getParamsNum(self):
        return sum(p.numel() for p in self.parameters())

class dimChecker(nn.Module):
    def __init__(self, dim, extendAt):
        super(dimChecker, self).__init__()
        self.dims = dim
        self.extendAt = extendAt
        self.inputShapeWarned = False

    def forward(self, x):
        if len(x.shape) != self.dims:
            if not self.inputShapeWarned:
                warnings.warn(f"Incorrect Input Shape {x.shape}, expected dim = {self.dims}")
                self.inputShapeWarned = True
            x = x.unsqueeze(self.extendAt)
        return x

class SCC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, embedded_size):
        super().__init__()
        kernel_conv2 = int(sfreq * ksize)
        kernel_pool = math.ceil(min(samples, sfreq * 0.5))
        self.conv1 = nn.Conv2d(1, spatial_channel, (channels, 1))

        pads = padding_same_tuple(kernel_conv2)
        self.pad2 = nn.ZeroPad2d(pads)
        self.conv2 = nn.Conv2d(spatial_channel, temporal_channel, (1, kernel_conv2))

        self.bn1 = nn.BatchNorm2d(spatial_channel)
        self.bn2 = nn.BatchNorm2d(temporal_channel)
        self.pool = nn.AvgPool2d((1, kernel_pool), (1, math.ceil(sfreq * 0.1)))
        self.Drop1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.pool(x)
        x = x + 1e-15
        x = torch.log(x)
        x = x.flatten(1)
        return x

class EEG(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, embedded_size):
        super().__init__()
        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.half_sf = math.floor(self.sf * ksize)

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        pads = padding_same_tuple(self.half_sf)
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(pads),
            nn.Conv2d(1, self.F1, (1, self.half_sf), bias=False),  # 62,32
            nn.BatchNorm2d(self.F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (self.ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        pads = padding_same_tuple(math.ceil(self.half_sf / 4))
        self.Conv3 = nn.Sequential(
            nn.ZeroPad2d(pads),
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, math.ceil(self.half_sf / 4)), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        return x

class SCCNetFC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, embedded_size):
        super().__init__()
        kernel_pool = math.ceil(min(samples, sfreq * 0.5))

        out = samples
        out = pool_out(out, kernel_pool, math.ceil(sfreq * 0.1))
        self.classifier = nn.Linear(temporal_channel * out, embedded_size)
        self.classifier_out = nn.Linear(temporal_channel * out, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        out = self.classifier_out(x)
        x = self.classifier(x)
        return x, out

class EEGNetStackFC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, embedded_size):
        super().__init__()
        F2 = 16

        out = samples
        out = pool_out(out, 4, 4)
        out = pool_out(out, 8, 8)
        eeg_embedded = F2 * out

        self.classifier = nn.Linear(eeg_embedded, embedded_size)
        self.classifier_out = nn.Linear(eeg_embedded, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        out = self.classifier_out(x)
        x = self.classifier(x)

        return x, out

class Stack(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size):
        super().__init__()
        self.stacks = nn.ModuleList([SCC(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, k, embedded_size) for k in ksize])
        self.stacks_fc = nn.ModuleList([SCCNetFC(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, k, embedded_size) for k in ksize])
        self.eeg_stacks = nn.ModuleList([EEG(n_classes, channels, samples, sfreq, k, embedded_size) for k in eeg_ksize])
        self.eeg_stacks_fc = nn.ModuleList([EEGNetStackFC(n_classes, channels, samples, sfreq, k, embedded_size) for k in eeg_ksize])
        self.klen = len(ksize) + len(eeg_ksize)

    def forward(self, x, **kwargs):
        x_output = []
        output = []
        for stack, fc in zip(self.stacks, self.stacks_fc):
            out = stack(x)
            x_out, out = fc(out)
            x_out = x_out.flatten(1)
            x_output.append(x_out)
            output.append(out)

        for stack, fc in zip(self.eeg_stacks, self.eeg_stacks_fc):
            out = stack(x)
            x_out, out = fc(out)
            x_out = x_out.flatten(1)
            x_output.append(x_out)
            output.append(out)

        # x_output = torch.stack(x_output, 1)
        # x_output = torch.sum(x_output, 1)
        x_output = torch.cat(x_output, 1)

        return x_output, output

class ClassificationHead(nn.Sequential):
    def __init__(self, n_classes, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size, tsne):
        super().__init__()
        #self.classifier = nn.Linear(embedded_size, n_classes)
        self.classifier = nn.Linear(embedded_size * (len(ksize) + len(eeg_ksize)), n_classes)
        self.Drop1 = nn.Dropout(0.4)

        self.tsne = tsne

    def forward(self, x):
        x_output, output = x
        x_output = self.Drop1(x_output)
        
        if self.tsne:
            features = x_output.view(x_output.size()[0], -1)
            return features
        
        x_output = self.classifier(x_output)
        return x_output

def MBSzEEGNet(channels, samples, sfreq=200, num_classes=2, tsne=False):
    return Multi(num_classes, channels, samples, sfreq, ksize=[0.1, 0.8], eeg_ksize=[0.1, 0.5], spatial_channel=channels, tsne=tsne)
