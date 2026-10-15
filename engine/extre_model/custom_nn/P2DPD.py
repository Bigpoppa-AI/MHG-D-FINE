import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        # [1,2,1]x[1,2,1] / 16
        filt = torch.tensor([[1., 2., 1.],
                             [2., 4., 2.],
                             [1., 2., 1.]])
        filt = filt / filt.sum()
        self.register_buffer('filt', filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)

class P2DownsampleDP(nn.Module):
 
    def __init__(self, c2, c3, cm=96):
        super().__init__()
        self.pre = ConvBNAct(c2, cm, k=3, s=1)           
        self.blur_down = BlurPool2d(cm, stride=2)       
        self.proj = ConvBNAct(cm, c3, k=1, s=1, p=0, act=False)  

        self.short = ConvBNAct(c2, c3, k=1, s=2, p=0, act=False) 
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, p2):
        x = self.pre(p2)
        x = self.blur_down(x)
        x = self.proj(x)

        r = self.short(p2)
        d2 = self.out_act(x + r)
        return d2
