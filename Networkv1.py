import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, UpsamplingBilinear2d, AvgPool2d, PReLU, Flatten, LayerNorm
from torch.nn import Module, ModuleList, Sequential
from torch.nn.utils import spectral_norm
from torch.optim import Adam

BETAS = (0, 0.99)

#constant
ROOT_2 = 1.41421
ln2 = 0.69314

def init_weights(m):
    if type(m) == Linear or type(m) == Conv2d:
        torch.nn.init.normal_(m.weight)
        #torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

def make_noise_img(size):
    noise = np.random.normal(loc=0, scale=1.0, size=size)
    return torch.as_tensor(noise, dtype=torch.float32)

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
    

class Weight_Scaling(Module):
    def __init__(self, fan_in):
        super().__init__()
        self.kaiming_const = float(ROOT_2/np.sqrt(fan_in))

    def forward(self, x):
        #return x
        return self.kaiming_const*x

class Disc_Conv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scaling_1 = Weight_Scaling(in_channels*3*3)
        self.conv_1 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_1 = PReLU()
        self.weight_scaling_2 = Weight_Scaling(in_channels*3*3)
        self.conv_2 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu_2 = PReLU()
        self.avgpool2d = AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.weight_scaling_1(x)
        x = self.conv_1(x)
        x = self.prelu_1(x)
        x = self.weight_scaling_2(x)
        x = self.conv_2(x)
        x = self.prelu_2(x)
        x = self.avgpool2d(x)
        return x

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
        self.weight_scaling_1 = Weight_Scaling(in_channels*1*1)
        self.q_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.k_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.v_conv1x1 = spectral_norm(Conv2d(in_channels, in_channels//div_num, kernel_size=1))
        self.weight_scaling_2 = Weight_Scaling((in_channels//div_num)*1*1)
        self.sa_conv1x1 = spectral_norm(Conv2d(in_channels//div_num, in_channels, kernel_size=1))
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, c, h, w = x.size()
        t = x
        x = x.contiguous()
        x = self.weight_scaling_1(x)
        q = self.q_conv1x1(x).view(batch_size, -1, h*w)
        k = self.k_conv1x1(x).view(batch_size, -1, h*w)
        v = self.v_conv1x1(x).view(batch_size, -1, h*w)

        attn_map = F.softmax(torch.bmm(q.permute(0,2,1), k), dim=-1)
        sa_map = torch.bmm(v, attn_map.permute(0,2,1))
        sa_map = self.weight_scaling_2(sa_map)
        sa_map = sa_map.contiguous()
        sa_map = self.sa_conv1x1(sa_map.view(batch_size, -1, h, w))
        return self.gamma*sa_map + t


class Discriminator(Module):
    def __init__(self, disc_first_channel, disc_last_size, disc_nonlocal_loc, disc_lr, img_size, device):
        super().__init__()
        if not(device == 'cpu' or 'cuda:' in device):
            assert Exception('invalid argument in Network2.Discriminator')
        self.module_list = ModuleList()
        in_channels = 3
        out_channels = disc_first_channel
        self.module_list.append(Weight_Scaling(in_channels*1*1))
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=disc_first_channel, kernel_size=1, stride=1))
        self.module_list.append(PReLU())
        in_size = img_size
        cnt = 0
        while True:
            cnt += 1
            in_channels = out_channels
            out_channels *= 2
            if in_size == disc_last_size:
                break
            self.module_list.append(Disc_Conv(in_channels, out_channels))
            if cnt == disc_nonlocal_loc:
                print('disc: non_local block inserted, in_size: ', in_size//2)
                self.module_list.append(Non_Local(out_channels))
            in_size //= 2
        self.module_list.append(Minibatch_Stddev())
        self.module_list.append(Weight_Scaling((in_channels+1)*3*3))
        self.module_list.append(Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
        self.module_list.append(PReLU())
        self.module_list.append(Weight_Scaling(in_channels*4*4))
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0))
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
        return x

class Generator_Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_size, use_gpu):
        super().__init__()
        self.use_gpu = use_gpu
        self.upsample_layer = UpsamplingBilinear2d(scale_factor=2)
        self.out_channels = out_channels
        self.weight_scaling_1 = Weight_Scaling(in_channels*kernel_size*kernel_size)
        self.conv_1 = spectral_norm(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.prelu_1 = PReLU()
        self.weight_scaling_2 = Weight_Scaling(out_channels*kernel_size*kernel_size)
        self.conv_2 = spectral_norm(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1))
        self.prelu_2 = PReLU()
        self.style_scaling = Weight_Scaling(style_size)
        self.style_affine_1 = spectral_norm(Linear(style_size, in_channels))
        self.noise_scaler_1 = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))
        self.style_affine_2 = spectral_norm(Linear(style_size, in_channels))
        self.noise_scaler_2 = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))

    def forward(self, content, style_base):
        batch_size = content.size(0)
        H = content.size(2)
        W = content.size(3)
        noise = make_noise_img((batch_size, 1, 2*H, 2*W))
        if self.use_gpu:
            with torch.cuda.device_of(content):
                noise = noise.cuda()
        else:
            noise = noise.cpu()
        content = self.upsample_layer(content)
        content = self.weight_scaling_1(content)
        content = self.prelu_1(self.conv_1(content) + self.noise_scaler_1*noise)
        content = AdaIN(content, self.style_affine_1(self.style_scaling(style_base)).view(-1, 2*self.out_channels, 1, 1))
        content = self.weight_scaling_2(content)
        content = self.prelu_2(self.conv_2(content) + self.noise_scaler_2*noise)
        content = AdaIN(content, self.style_affine_2(self.style_scaling(style_base)).view(-1, 2*self.out_channels, 1, 1))
        return content


class StyleMapper(Module):
    def __init__(self, z_size, style_size, mapping_lr, device):
        super().__init__()
        if not(device == 'cpu' or 'cuda:' in device):
            assert Exception('invalid argument in Network1.StyleMapper')
        self.styleblock = Sequential(
            Weight_Scaling(z_size),
            spectral_norm(Linear(z_size, style_size//4)),
            PReLU(),
            Weight_Scaling(style_size//4),
            spectral_norm(Linear(style_size//4, style_size//2)),
            PReLU(),
            Weight_Scaling(style_size//2),
            spectral_norm(Linear(style_size//2, style_size)),
            PReLU(),
            Weight_Scaling(style_size),
            spectral_norm(Linear(style_size, style_size)),
        )
        self.to(device)
        self.opt = Adam(self.parameters(), lr=mapping_lr, betas=BETAS)
        self.apply(init_weights)


    def forward(self, z):
        style_base = self.styleblock(z)
        return style_base

class Generator(Module):
    def __init__(self, gen_channel, texture_size, style_size, gen_nonlocal_loc, gen_lr, img_size, device):
        super().__init__()
        if device == 'cpu':
            self.use_gpu = False
        elif 'cuda:' in device:
            self.use_gpu = True
        else:
            assert Exception('invalid argument in Network2.Generator')
        self.gen_channel = gen_channel
        self.basic_texture = torch.nn.Parameter(torch.rand(gen_channel, texture_size, texture_size))
        self.weight_scaling_1 = Weight_Scaling(gen_channel*3*3)
        self.conv = spectral_norm(Conv2d(in_channels=gen_channel, out_channels=gen_channel, kernel_size=3, stride=1, padding=1))
        self.prelu = PReLU()
        self.style_scaling = Weight_Scaling(style_size)
        self.style_affine_1 = spectral_norm(Linear(style_size, gen_channel*2))
        self.noise_scaler_1 = torch.nn.Parameter(torch.zeros(gen_channel).view(1, gen_channel, 1, 1))
        self.style_affine_2 = spectral_norm(Linear(style_size, gen_channel*2))
        self.noise_scaler_2 = torch.nn.Parameter(torch.zeros(gen_channel).view(1, gen_channel, 1, 1))
        self.module_list = ModuleList()
        in_channels = gen_channel
        in_size = 4
        cnt = 0
        while True:
            cnt += 1
            self.module_list.append(Generator_Conv(in_channels, in_channels//2, 3, style_size, self.use_gpu))
            if cnt == gen_nonlocal_loc:
                print('gen: non_local block inserted, in_size: ', 2*in_size)
                self.module_list.append(Non_Local(in_channels//2))
            in_channels //= 2
            in_size *= 2
            if in_size >= img_size:
                break
        self.weight_scaling_2 = Weight_Scaling(in_channels*1*1)
        self.last_layer = spectral_norm(Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, stride=1))
        self.to(device)
        self.opt = Adam(self.parameters(), lr=gen_lr, betas=BETAS)
        self.apply(init_weights)


    def forward(self, style_base):
        batch_size = style_base.size()[0]
        x = self.basic_texture.repeat(batch_size, 1, 1, 1)
        noise = make_noise_img((batch_size, 1, 4, 4))
        if self.use_gpu:
            with torch.cuda.device_of(style_base):
                noise = noise.cuda()
        else:
            noise = noise.cpu()
        x = x + self.noise_scaler_1*noise
        x = AdaIN(x, self.style_affine_1(self.style_scaling(style_base)).view(-1, 2*self.gen_channel, 1, 1))
        x = self.prelu(self.conv(self.weight_scaling_1(x)) + self.noise_scaler_2*noise)
        x = AdaIN(x, self.style_affine_2(self.style_scaling(style_base)).view(-1, 2*self.gen_channel, 1, 1))
        for m in self.module_list:
            if type(m) != Non_Local:
                x = m(x, style_base)
            else:
                x = m(x)
        x = self.weight_scaling_2(x)
        #fucking bug! pytorch1.4.0 has bug on conv with 1*1 kernel!
        x = x.contiguous()
        #above line fix the bug
        x = self.last_layer(x)
        x = torch.clamp(x, min=0, max=1)
        return x

if __name__ == '__main__':
    print('testing Networkv1.py')