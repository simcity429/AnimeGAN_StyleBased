import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import Networkv1
import Networkv2
from copy import deepcopy
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.nn import Module, DataParallel
from CustomDataset import TANOCIv2_Dataset

COMPUTE_MEAN_ITER = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_num', required=True)
    parser.add_argument('--load_index', required=True)
    parser.add_argument('--load_path', default='./result')
    parser.add_argument('--save_path', default='./sample/')
    parser.add_argument('--psi', default=0.5)
    parser.add_argument('--version', default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--img_level', default=7)
    parser.add_argument('--batch_size', default=24)
    parser.add_argument('--z_cov', default=1)
    parser.add_argument('--interpolate_num', default=8)
    parser.add_argument('--style_size', default=128)
    parser.add_argument('--z_size', default=8)
    parser.add_argument('--texture_size', default=4)
    parser.add_argument('--gen_channel', default=256)
    parser.add_argument('--gen_nonlocal_loc', default=-1)
    parser.add_argument('--disc_first_channel', default=16)
    parser.add_argument('--disc_last_size', default=4)
    parser.add_argument('--disc_nonlocal_loc', default=2)

    args = parser.parse_args()
    gen_num = int(args.gen_num)
    load_index = int(args.load_index)
    if load_index <= 0:
        raise Exception('You have to specify load_index')
    load_path = str(args.load_path)
    save_path = str(args.save_path)
    psi = float(args.psi)
    version = int(args.version)
    device = str(args.device)

    img_level = int(args.img_level)
    img_size = 2**img_level
    batch_size = int(args.batch_size)
    z_cov = float(args.z_cov)
    interpolate_num = int(args.interpolate_num)

    style_size = int(args.style_size)
    z_size = int(args.z_size)
    texture_size = int(args.texture_size)
    gen_channel = int(args.gen_channel)
    gen_nonlocal_loc = int(args.gen_nonlocal_loc)

    disc_first_channel = int(args.disc_first_channel)
    disc_last_size = int(args.disc_last_size)
    disc_nonlocal_loc = int(args.disc_nonlocal_loc)

    if device == 'multi':
        raise Exception('multi gpu not allowed in sampling!')
    elif device == 'cpu':
        use_multi_gpu = False
    elif 'cuda' in device:
        use_multi_gpu = False
    else:
        raise Exception('invalid argument in device (sampling)')

    S = Networkv1.StyleMapper(z_size, style_size, 0, device)
    if version == 1:
        G = Networkv1.Generator(gen_channel, texture_size, style_size, gen_nonlocal_loc, 0, img_size, device)
    elif version == 2:
        G = Networkv2.Generator(gen_channel, texture_size, style_size, gen_nonlocal_loc, 0, img_size, device)
    else:
        raise Exception('invalid version')

    print('loading', load_index, 'models...')
    S_load_path = load_path + '_weight/' + str(load_index) + 'S.pt'
    G_load_path = load_path + '_weight/' + str(load_index) + 'G.pt'
    S.load_state_dict(torch.load(S_load_path))
    G.load_state_dict(torch.load(G_load_path))
    dist = MultivariateNormal(loc=torch.zeros(batch_size, z_size), covariance_matrix=z_cov*torch.eye(z_size))
    tmp = None
    for _ in range(COMPUTE_MEAN_ITER):
        z = dist.sample()
        if 'cuda' in device:
            z = z.cuda()
        w = S(z)
        if tmp is None:
            tmp = w
        else:
            tmp += w
    style_mean = torch.mean(tmp, axis=0)/COMPUTE_MEAN_ITER
    
    for e in range(gen_num):
        z = dist.sample()
        if 'cuda' in device:
            z = z.cuda()
        w = S(z)
        w = style_mean + psi*(w - style_mean)
        img = G(w).detach()
        img_save_path = save_path + str(e) + '.jpg'
        save_image(make_grid(img), img_save_path, normalize=True)
    



    



