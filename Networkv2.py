import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, UpsamplingBilinear2d, AvgPool2d, PReLU, Flatten, LayerNorm
from torch.nn import Module, ModuleList, Sequential
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from Networkv1 import make_noise_img, build_disc_convblock, StyleMapper, Non_Local, Minibatch_Stddev
from config import *


def init_weights(m):
    if type(m) == Linear or type(m) == Conv2d:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.disc_block = build_disc_convblock(in_channels, out_channels)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        t = x
        x = self.disc_block(x)
        t = t.contiguous()
        t = F.avg_pool2d(self.conv(t))
        x = (x + t)/ROOT_2
        return x

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.module_list = ModuleList()
        in_channels = 3
        out_channels = DISC_FIRST_CHANNEL
        #fromRGB
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=DISC_FIRST_CHANNEL, kernel_size=1, stride=1))
        in_size = IMG_SIZE
        cnt = 0
        while True:
            cnt += 1
            in_channels = out_channels
            out_channels *= 2
            if in_size == DISC_LAST_SIZE:
                break
            self.module_list.append(build_disc_convblock(in_channels, out_channels))
            if cnt in DISC_NON_LOCAL_LOC_V2:
                print('disc: non_local block inserted')
                self.module_list.append(Non_Local(out_channels))
            in_size //= 2
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0))
        self.module_list.append(PReLU())
        self.module_list.append(Flatten())
        self.module_list.append(Linear(in_channels, 1))
        self.opt = Adam(self.parameters(), lr=DISC_LR, betas=DISC_BETAS)
        self.apply(init_weights)
        self.to(DEVICE)

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
            if (x != x).any():
                print('NaN occur!')
                assert False
        return x

class _ModulatedConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.prelu = PReLU()
        #[Cout,Cin,k,k]

    def forward(self, x, style_std):
        # x: [N,Cin,H,W]
        # style_std: [N,Cin]
        batch_size = x.size(0)
        in_channels = x.size(1)
        out_channels = self.weight.size(0)
        H = x.size(2)
        W = x.size(3)
        weight = self.weight.view(1, self.weight.size(0), self.weight.size(1), self.weight.size(2), self.weight.size(3))
        #[1,Cout,Cin,k,k]*[batch,1,Cin,1,1]
        weight = weight*(style_std.view(style_std.size(0), 1, style_std.size(1), 1, 1))
        #[batch,Cout,Cin,k,k]
        weight_l2 = torch.sqrt(torch.sum(weight**2, dim=(2,3,4), keepdim=True)+1e-8)
        weight = weight/weight_l2
        weight = weight.view(-1,weight.size(2), weight.size(3), weight.size(4))
        #[batch*Cout,Cin,H,W]
        x = x.view(1, -1, x.size(2), x.size(3))
        #[1,N*C,H,W]
        x = F.conv2d(x, weight, groups=batch_size, padding=1)
        x = x.view(batch_size, out_channels, H, W) + self.bias
        x = self.prelu(x)
        return x

class ModulatedConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, up=False, out=False):
        #up: upsamplex2?
        #out: RGB out?
        super().__init__()
        self.up = up
        self.style_affine = Linear(STYLE_SIZE, in_channels)
        self.modulated_conv = _ModulatedConv(in_channels, out_channels, kernel_size)
        self.noise_scalar = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))
        if out:
            self.name = 'LATTER'
            self.out = True
            self.out = Conv2d(out_channels, 3, 1)
        else:
            self.name = 'FORMER'
            self.out = False

    def forward(self, x, style_base):
        #x: [N,C,H,W]
        #style_base: [N,STYLE_SIZE]
        batch_size = x.size(0)
        style_std = self.style_affine(style_base)+1
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        img_size = x.size(2)
        noise = make_noise_img(batch_size, img_size)
        x = self.modulated_conv(x, style_std)
        x = x + self.noise_scalar*noise
        if self.out:
            x = x.contiguous()
            out = self.out(x)
            out = torch.clamp(out, min=0, max=1)
            return x, out
        else:
            return x

class Generator(Module):
    def __init__(self):
        super().__init__()
        self.basic_texture = torch.nn.Parameter(torch.rand(GEN_CHANNEL, TEXTURE_SIZE, TEXTURE_SIZE))
        self.module_list = ModuleList()
        first_block = ModulatedConvBlock(GEN_CHANNEL, GEN_CHANNEL//2, kernel_size=3, up=False, out=True)
        self.module_list.append(first_block)
        in_size = 2*TEXTURE_SIZE
        in_channels = GEN_CHANNEL//2
        cnt = 0
        while True:
            cnt += 1
            former = ModulatedConvBlock(in_channels, in_channels, kernel_size=3, up=True, out=False)
            latter = ModulatedConvBlock(in_channels, in_channels//2, kernel_size=3, up=False, out=True)
            self.module_list.append(former)
            self.module_list.append(latter)
            if cnt in GEN_NON_LOCAL_LOC_V2:
                print('gen: non_local block inserted')
                self.module_list.append(Non_Local(in_channels//2))
            in_size *= 2
            in_channels //= 2
            if in_size > IMG_SIZE:
                break
        self.opt = Adam(self.parameters(), lr=GEN_LR, betas=GEN_BETAS)
        self.apply(init_weights)
        self.to(DEVICE)

    def forward(self, style_base):
        img = None
        batch_size = style_base.size(0)
        x = self.basic_texture.repeat(batch_size, 1, 1, 1)
        for m in self.module_list:
            if m.name == 'FORMER':
                x = m(x, style_base)
            elif m.name == 'LATTER':
                x, rgb = m(x, style_base)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == IMG_SIZE:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
            elif m.name == 'NON_LOCAL':
                x = m(x)
            else:
                raise NotImplementedError(m.name,'what are you doing!')
        img = torch.clamp(img, min=0, max=1)
        #img = torch.sigmoid(img)
        return img



if __name__ == '__main__':
    print('testing Networkv2.py')
    
