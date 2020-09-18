""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def apply_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name,param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
            
class RegressorUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_reduce,n_hidden,n_regress,img_size,bilinear=True):
        super(RegressorUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_reduce  = n_reduce
        self.n_hidden  = n_hidden
        self.img_size  = img_size
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sm = nn.Softmax(dim=1)
        
        self.reduce = Reducer(n_classes,n_reduce)
        
        in_channels = int((n_classes if not n_reduce else n_reduce[-1])*(img_size/(2**len(n_reduce)))**2)
        
        self.mlp = MLP(in_channels, n_regress, n_hidden)
            
        
    def apply_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name,param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        x6 = self.sm(logits)
        
        x7 = self.reduce(x6)
        shp = x7.shape
        xb  = x7.view(shp[0],shp[1]*shp[2]*shp[3])
        regs = self.mlp(xb)
        
        return regs
    
# class RegressorUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, n_regress,n_hidden,img_size, n_bottle,bilinear=True):
#         super(RegressorUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.n_regress = n_regress
#         self.n_hidden  = n_hidden
#         self.img_size  = img_size
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
        
# #         self.reduce = DoubleConv(1024 // factor, n_bottle,mid_channels=2*n_bottle)
# #         bnsize = int((n_bottle)*(img_size/(2**4))**2)
# #         self.mlp = MLP(bnsize,n_regress,n_hidden)
        
#         self.reduce = SingleConv(64,16)
#         bnsize = int(16*(img_size)**2)
#         self.mlp = MLP(bnsize,n_regress,n_hidden)
        
#     def apply_state_dict(self, state_dict):
#         own_state = self.state_dict()
#         for name,param in state_dict.items():
#             if name not in own_state:
#                 continue
# #             if isinstance(param,Parameter):
# #                 param = param.data
#             own_state[name].copy_(param)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
        
# #         x6  = self.reduce(x5)
# #         shp = x6.shape
# #         xb  = x6.view(shp[0],shp[1]*shp[2]*shp[3])
# #         regs = self.mlp(xb)
        
#         x6 = self.reduce(x1)
#         shp = x6.shape
#         xb  = x6.view(shp[0],shp[1]*shp[2]*shp[3])
#         regs = self.mlp(xb)
        
#         return logits, regs


# class RegressorNet(nn.Module):
#     def __init__(self, n_channels,n_regress,n_hidden,img_size):
#         super(RegressorNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_regress = n_regress
#         self.n_hidden  = n_hidden
#         self.img_size  = img_size

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
        
# #         self.reduce = SingleConv(512,n_bottle)
# #         bnsize = int(n_bottle*(img_size/(2**4))**2)
#         bnsize = int(512*(img_size/(2**4))**2)
#         self.mlp = MLP(bnsize,n_regress,n_hidden)
        
#     def apply_state_dict(self, state_dict):
#         own_state = self.state_dict()
#         for name,param in state_dict.items():
#             if name not in own_state:
#                 continue
#             own_state[name].copy_(param)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)        
# #         x6 = self.reduce(x5)
# #         shp = x6.shape
# #         xb  = x6.view(shp[0],shp[1]*shp[2]*shp[3])
#         shp = x5.shape
#         xb = x5.view(shp[0],shp[1]*shp[2]*shp[3])
#         regs = self.mlp(xb)
        
#         return regs