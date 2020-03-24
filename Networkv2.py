import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, UpsamplingBilinear2d, AvgPool2d, PReLU, Flatten, LayerNorm
from torch.nn import Module, ModuleList, Sequential
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from Networkv1 import make_noise_img, Weight_Scaling, Disc_Conv, StyleMapper, Non_Local, Minibatch_Stddev

BETAS = (0, 0.99)

#constant
ROOT_2 = 1.41421
ln2 = 0.69314

def init_weights(m):
    if type(m) == Linear or type(m) == Conv2d or type(m) == _ModulatedConv:
        torch.nn.init.normal_(m.weight)
        #torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.disc_block = Disc_Conv(in_channels, out_channels)
        self.weight_scaling = Weight_Scaling(in_channels*3*3)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        t = x
        x = self.disc_block(x)
        t = t.contiguous()
        t = F.avg_pool2d(self.conv(self.weight_scaling(t)), kernel_size=2, stride=2)
        x = (x + t)/ROOT_2
        return x

class Discriminator(Module):
    def __init__(self, disc_first_channel, disc_last_size, disc_nonlocal_loc, disc_lr, img_size, device):
        super().__init__()
        if not(device == 'cpu' or 'cuda:' in device):
            assert Exception('invalid argument in Network2.Discriminator')
        self.module_list = ModuleList()
        in_channels = 3
        out_channels = disc_first_channel
        #fromRGB
        self.module_list.append(Weight_Scaling(in_channels*1*1))
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=disc_first_channel, kernel_size=1, stride=1))
        in_size = img_size
        cnt = 0
        while True:
            cnt += 1
            in_channels = out_channels
            out_channels *= 2
            if in_size == disc_last_size:
                break
            self.module_list.append(ResidualBlock(in_channels, out_channels))
            if cnt == disc_nonlocal_loc:
                print('disc: non_local block inserted, in_size: ', in_size//2)
                self.module_list.append(Non_Local(out_channels))
            in_size //= 2
        self.module_list.append(Minibatch_Stddev())
        self.module_list.append(Weight_Scaling((in_channels+1)*4*4))
        self.module_list.append(Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=4, stride=1, padding=0))
        self.module_list.append(PReLU())
        self.module_list.append(Flatten())
        self.module_list.append(Weight_Scaling(in_channels))
        self.module_list.append(Linear(in_channels, 1))
        self.to(device)
        self.opt = Adam(self.parameters(), lr=disc_lr, betas=BETAS)
        self.apply(init_weights)

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

    def forward(self, x, style_std, noise):
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
        padding_size = (self.weight.size(3)-1)//2
        x = F.conv2d(x, weight, groups=batch_size, padding=padding_size)
        x = x.view(batch_size, out_channels, H, W) + self.bias
        x += noise
        x = self.prelu(x)
        return x

class ModulatedConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_size, use_gpu, up, out):
        #up: upsamplex2?
        #out: RGB out?
        super().__init__()
        self.use_gpu = use_gpu
        self.up = up
        self.style_scaling = Weight_Scaling(style_size)
        self.style_affine = Linear(style_size, in_channels)
        self.modulated_conv = _ModulatedConv(in_channels, out_channels, kernel_size)
        self.noise_scalar = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))
        if out:
            self.name = 'LATTER'
            self.out = True
            self.out_weight_scale = Weight_Scaling(out_channels*1*1)
            self.out_conv = Conv2d(out_channels, 3, 1)
        else:
            self.name = 'FORMER'
            self.out = False

    def forward(self, x, style_base, t=None):
        #x: [N,C,H,W]
        #style_base: [N,STYLE_SIZE]
        #t: for 'LATTER' block, residual connection!
        batch_size = x.size(0)
        style_std = self.style_affine(self.style_scaling(style_base))+1
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        img_size = x.size(2)
        noise = make_noise_img((batch_size, 1, img_size, img_size))
        if self.use_gpu:
            with torch.cuda.device_of(x):
                noise = noise.cuda()
        else:
            noise = noise.cpu()
        x = self.modulated_conv(x, style_std, self.noise_scalar*noise)
        if self.out:
            x = x.contiguous()
            if t is not None:
                x += t
                x /= ROOT_2
            out = self.out_conv(self.out_weight_scale(x))
            out = torch.clamp(out, min=0, max=1)
            return x, out
        else:
            return x

class Generator(Module):
    def __init__(self, gen_channel, texture_size, style_size, gen_nonlocal_loc, gen_lr, img_size, device):
        super().__init__()
        if device == 'cpu':
            use_gpu = False
        elif 'cuda:' in device:
            use_gpu = True
        else:
            assert Exception('invalid argument in Network2.Generator')
        self.img_size = img_size
        self.basic_texture = torch.nn.Parameter(torch.normal(torch.zeros(gen_channel, texture_size, texture_size), 1.0))
        self.module_list = ModuleList()
        self.conv1x1_list = ModuleList()
        first_block = ModulatedConvBlock(gen_channel, gen_channel, 3, style_size, use_gpu, up=False, out=True)
        self.module_list.append(first_block)
        in_size = 2*texture_size
        in_channels = gen_channel
        cnt = 0
        while True:
            cnt += 1
            former = ModulatedConvBlock(in_channels, in_channels, 3, style_size, use_gpu, up=True, out=False)
            if cnt > 1:
                latter = ModulatedConvBlock(in_channels, in_channels//2, 3, style_size, use_gpu, up=False, out=True)
                conv1x1 = Conv2d(in_channels, in_channels//2, 1)
                out_channels = in_channels//2
            else:
                latter = ModulatedConvBlock(in_channels, in_channels, 3, style_size, use_gpu, up=False, out=True)
                conv1x1 = Conv2d(in_channels, in_channels, 1)
                out_channels = in_channels
            self.module_list.append(former)
            self.module_list.append(latter)
            self.conv1x1_list.append(conv1x1)
            if cnt == gen_nonlocal_loc:
                print('gen: non_local block inserted, in_size: ', 2*in_size)
                self.module_list.append(Non_Local(out_channels))
            in_size *= 2
            in_channels = out_channels
            if in_size > img_size:
                break
        self.to(device)
        self.opt = Adam(self.parameters(), lr=gen_lr, betas=BETAS)
        self.apply(init_weights)

    def forward(self, style_base):
        img = None
        cnt = 0
        batch_size = style_base.size(0)
        x = self.basic_texture.repeat(batch_size, 1, 1, 1)
        t = None
        # t is for residual connection between 'FORMER' block and 'LATTER' block
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                t /= self.conv1x1_list[cnt-1].weight.size(1)
                x = m(x, style_base)
            elif m.name == 'LATTER':
                cnt += 1
                x, rgb = m(x, style_base, t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
            elif m.name == 'NON_LOCAL':
                x = m(x)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        img = torch.clamp(img, min=0, max=1)
        return img



if __name__ == '__main__':
    print('testing Networkv2.py')
    
