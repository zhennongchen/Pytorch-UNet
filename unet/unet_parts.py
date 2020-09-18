""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationRegressionLoss(nn.Module):
    
    def __init__(self):
        super(SegmentationRegressionLoss,self).__init__()
    
    def forward(self, labels, vector, target_labels, target_vector, weight):
        CE = nn.CrossEntropyLoss()(labels,target_labels)
        MSE = nn.MSELoss()(vector,target_vector)
        L = (CE + weight*MSE).double()
        
        print("CE:  " + str(CE))
        print("MSE: " + str(MSE))
        print("L:   " + str(L))
        
        return L

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    
    
class Reducer(nn.Module):
    def __init__(self,n_classes,n_reduce):
        super(Reducer, self).__init__()
        if not n_reduce:
            self.reduce = Identity()
        else:
            modules = []
            modules.append(SingleConv(n_classes, n_reduce[0]))
#             modules.append(nn.BatchNorm2d(n_reduce[0]))
#             modules.append(nn.ReLU(inplace=True))
            modules.append(nn.MaxPool2d(2))
            
            if len(n_reduce) > 1:
                for ind in range(len(n_reduce)-1):
                    modules.append(SingleConv(n_reduce[ind],n_reduce[ind+1]))
                    modules.append(nn.BatchNorm2d(n_reduce[ind+1]))
                    modules.append(nn.ReLU(inplace=True))
                    modules.append(nn.MaxPool2d(2))
            
            self.reduce = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.reduce(x)
    
class MLP(nn.Module):
    """(fully connected)"""
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_channels,hidden_channels[0]))
        modules.append(nn.ReLU())
        
        Nhidden = len(hidden_channels)
        
        if Nhidden > 1:
            for hiddenInd in range(Nhidden-1):
                modules.append(nn.Linear(hidden_channels[hiddenInd],hidden_channels[hiddenInd+1]))
                modules.append(nn.ReLU())
        
        modules.append(nn.Linear(hidden_channels[-1],out_channels))
        
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)
    
# class MLP(nn.Module):
#     """(fully connected)"""
    
#     def __init__(self, in_channels, out_channels, hidden_channels):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels,hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels,out_channels),
#         )

#     def forward(self, x):
#         return self.mlp(x)

# class MLP(nn.Module):
#     """(fully connected)"""
    
#     def __init__(self, in_channels, out_channels, hidden_channels):
#         super().__init__()
#         modules = []
#         modules.append(nn.Linear(in_channels,hidden_channels[0]))
#         modules.append(nn.ReLU())
        
#         Nhidden = len(hidden_channels)
        
#         if Nhidden > 1:
#             for hiddenInd in range(Nhidden-1):
#                 modules.append(nn.Linear(hidden_channels[hiddenInd],hidden_channels[hiddenInd+1]))
#                 modules.append(nn.ReLU())
        
#         modules.append(nn.Linear(hidden_channels[-1],out_channels))
        
#         self.mlp = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.mlp(x)
