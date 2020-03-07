import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, UpsamplingBilinear2d, AvgPool2d, PReLU, Flatten, LayerNorm
from torch.nn import Module, ModuleList, Sequential
from torch.nn.utils import spectral_norm
from torch.optim import Adam
from config import *

def init_weights(m):
    if type(m) == Linear or type(m) == Conv2d:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

def make_noise_img(batch_size, size):
    noise = np.random.normal(loc=0.5, scale=0.3, size=size)
    noise = np.where(noise > 1, 1, noise)
    noise = np.where(noise < 0, 0, noise)
    noise = np.tile(noise, (batch_size, 1, 1, 1))
    return torch.as_tensor(noise, dtype=torch.float32, device=DEVICE)

def AdaIN(content, style):
    #content: (N,C,H,W) torch.FloatTensor
    #style: (N,2*C,1,1) torch FloatTensor
    assert len(content.size()) == 4
    assert len(style.size()) == 4
    C = content.size()[1]
    content_mean = torch.mean(content, dim=(2,3), keepdim=True)
    content_std = torch.std(content, dim=(2,3), keepdim=True)
    style_mean = style[:, :C, :, :]
    style_std = style[:, C:, :, :]
    out = (style_std + 1)*((content - content_mean)/content_std) + style_mean
    return out
    
def build_disc_convblock(in_channels, out_channels):
     return Sequential(
         Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
         PReLU(),
         Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
         PReLU(),
         AvgPool2d(kernel_size=2, stride=2),
     )


class Minibatch_Stddev(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #x: (N, C, H, W) torch.FloatTensor
        assert len(x.size()) == 4
        batch_size = x.size(0)
        H = x.size(2)
        W = x.size(3)
        stat = torch.std(x, dim=0)
        #(C, H, W)
        stat = torch.mean(stat, dim=0)
        #(H, W)
        stat = stat.repeat(batch_size, 1, 1, 1)
        x = torch.cat((x, stat), dim=1)
        return x


class Non_Local(Module):
    def __init__(self, in_channels, div_num=4):
        super().__init__()
        assert in_channels % div_num == 0, "The remainder of 'in_ch/div_num' must be zero."
        self.name = 'NON_LOCAL'
        self.q_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.k_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.v_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.sa_conv1x1 = spectral_norm(Conv2d(in_channels//div_num, in_channels, kernel_size=1))
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, c, h, w = x.size()
        x = x.contiguous()
        q = self.q_conv1x1(x).view(batch_size, -1, h*w)
        k = self.k_conv1x1(x).view(batch_size, -1, h*w)
        v = self.v_conv1x1(x).view(batch_size, -1, h*w)

        attn_map = F.softmax(torch.bmm(q.permute(0,2,1), k), dim=-1)
        sa_map = torch.bmm(v, attn_map.permute(0,2,1))
        sa_map = sa_map.contiguous()
        sa_map = self.sa_conv1x1(sa_map.view(batch_size, -1, h, w))
        return self.gamma*sa_map + x


class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.module_list = ModuleList()
        in_channels = 3
        out_channels = DISC_FIRST_CHANNEL
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=DISC_FIRST_CHANNEL, kernel_size=1, stride=1))
        self.module_list.append(PReLU())
        in_size = IMG_SIZE
        cnt = 0
        while True:
            cnt += 1
            in_channels = out_channels
            out_channels *= 2
            if in_size == DISC_LAST_SIZE:
                break
            self.module_list.append(build_disc_convblock(in_channels, out_channels))
            if cnt in DISC_NON_LOCAL_LOC_V1:
                print('disc: non_local block inserted')
                self.module_list.append(Non_Local(out_channels))
            in_size //= 2
        self.module_list.append(Minibatch_Stddev())
        self.module_list.append(Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
        self.module_list.append(PReLU())
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
        return x

class Generator_Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.upsample_layer = UpsamplingBilinear2d(scale_factor=2)
        self.out_channels = out_channels
        self.conv_1 = spectral_norm(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.prelu_1 = PReLU()
        self.conv_2 = spectral_norm(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.prelu_2 = PReLU()
        self.style_affine_1 = spectral_norm(Linear(STYLE_SIZE, in_channels))
        self.noise_scaler_1 = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))
        self.style_affine_2 = spectral_norm(Linear(STYLE_SIZE, in_channels))
        self.noise_scaler_2 = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))

    def forward(self, content, style_base):
        batch_size = content.size(0)
        H = content.size(2)
        W = content.size(3)
        noise = make_noise_img(batch_size, (2*H, 2*W))
        content = self.upsample_layer(content)
        content = self.prelu_1(self.conv_1(content))
        content = content + self.noise_scaler_1*noise
        content = AdaIN(content, self.style_affine_1(style_base).view(-1, 2*self.out_channels, 1, 1))
        content = self.prelu_2(self.conv_2(content))
        content = content + self.noise_scaler_2*noise
        content = AdaIN(content, self.style_affine_2(style_base).view(-1, 2*self.out_channels, 1, 1))
        return content


class StyleMapper(Module):
    def __init__(self):
        super().__init__()
        self.styleblock = Sequential(
            spectral_norm(Linear(Z_SIZE, STYLE_SIZE//4)),
            PReLU(),
            spectral_norm(Linear(STYLE_SIZE//4, STYLE_SIZE//2)),
            PReLU(),
            spectral_norm(Linear(STYLE_SIZE//2, STYLE_SIZE)),
            PReLU(),
            spectral_norm(Linear(STYLE_SIZE, STYLE_SIZE)),
        )
        self.opt = Adam(self.parameters(), lr=MAPPING_LR, betas=GEN_BETAS)
        self.apply(init_weights)
        self.to(DEVICE)


    def forward(self, z):
        style_base = self.styleblock(z)
        return style_base

class Generator(Module):
    def __init__(self):
        super().__init__()
        self.basic_texture = torch.nn.Parameter(torch.rand(GEN_CHANNEL, TEXTURE_SIZE, TEXTURE_SIZE))
        self.conv = spectral_norm(Conv2d(in_channels=GEN_CHANNEL, out_channels=GEN_CHANNEL, kernel_size=3, stride=1, padding=1))
        self.prelu = PReLU()
        self.style_affine_1 = spectral_norm(Linear(STYLE_SIZE, GEN_CHANNEL*2))
        self.noise_scaler_1 = torch.nn.Parameter(torch.zeros(GEN_CHANNEL).view(1, GEN_CHANNEL, 1, 1))
        self.style_affine_2 = spectral_norm(Linear(STYLE_SIZE, GEN_CHANNEL*2))
        self.noise_scaler_2 = torch.nn.Parameter(torch.zeros(GEN_CHANNEL).view(1, GEN_CHANNEL, 1, 1))
        self.module_list = ModuleList()
        in_channels = GEN_CHANNEL
        in_size = 4
        cnt = 0
        while True:
            cnt += 1
            self.module_list.append(Generator_Conv(in_channels, in_channels//2, 3))
            if cnt in GEN_NON_LOCAL_LOC_V1:
                print('gen: non_local block inserted')
                self.module_list.append(Non_Local(in_channels//2))
            in_channels //= 2
            in_size *= 2
            if in_size >= IMG_SIZE:
                break
        self.last_layer = spectral_norm(Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, stride=1))
        self.opt = Adam(self.parameters(), lr=GEN_LR, betas=GEN_BETAS)
        self.apply(init_weights)
        self.to(DEVICE)


    def forward(self, style_base):
        batch_size = style_base.size()[0]
        x = self.basic_texture.repeat(batch_size, 1, 1, 1)
        noise = make_noise_img(batch_size, 4)
        x = x + self.noise_scaler_1*noise
        x = AdaIN(x, self.style_affine_1(style_base).view(-1, 2*GEN_CHANNEL, 1, 1))
        x = self.prelu(self.conv(x))
        x = x + self.noise_scaler_2*noise
        x = AdaIN(x, self.style_affine_2(style_base).view(-1, 2*GEN_CHANNEL, 1, 1))
        for m in self.module_list:
            if type(m) != Non_Local:
                x = m(x, style_base)
            else:
                x = m(x)
        #fucking bug! pytorch1.4.0 has bug on conv with 1*1 kernel!
        x = x.contiguous()
        #above line fix the bug
        x = self.last_layer(x)
        x = torch.clamp(x, min=0, max=1)
        return x

if __name__ == '__main__':
    print('testing Networkv1.py')