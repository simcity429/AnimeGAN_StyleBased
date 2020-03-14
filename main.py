import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import Networkv1
import Networkv2
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.nn import BCEWithLogitsLoss, DataParallel
from CustomDataset import TANOCIv2_Dataset

#constant
ROOT_2 = 1.41421
ln2 = 0.69314

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #general arguments
    parser.add_argument('--version', default=2)
    parser.add_argument('--lir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--epoch', default=1000)
    parser.add_argument('--img_level', default=6)
    parser.add_argument('--batch_size', default=24)
    parser.add_argument('--verbose_freq', default=100)
    parser.add_argument('--save_freq', default=10)
    parser.add_argument('--z_cov', default=1)
    parser.add_argument('--pl_cov', default=0.1)
    parser.add_argument('--interpolate_num', default=8)
    parser.add_argument('--ema_decay', default=0.99)

    #generator arguments
    parser.add_argument('--style_size', default=64)
    parser.add_argument('--z_size', default=8)
    parser.add_argument('--texture_size', default=4)
    parser.add_argument('--gen_channel', default=256)
    parser.add_argument('--gen_nonlocal_loc', default=2)
    parser.add_argument('--gen_lr', default=0.0001)
    parser.add_argument('--mapping_lr', default=0.00001)
    parser.add_argument('--gen_lazy', default=8)

    #discriminator arguments
    parser.add_argument('--disc_first_channel', default=16)
    parser.add_argument('--disc_last_size', default=4)
    parser.add_argument('--disc_nonlocal_loc', default=2)
    parser.add_argument('--disc_lr', default=0.0004)
    parser.add_argument('--gp_coef', default=10)
    args = parser.parse_args()

    version = int(args.version)
    if args.lir == "True" or args.lir == "true":
        lir = True
    elif args.lir == "False" or args.lir == "false":
        lir = False
    else:
        raise Exception('wrong value on lir')
    device = str(args.device)
    #device: 'multi', 'cpu', 'cuda:%d'
    #'multi' uses all the GPUs
    dataset_path = str(args.dataset_path)
    save_path = str(args.save_path)
    epoch = int(args.epoch)
    img_level = int(args.img_level)
    img_size = 2**img_level
    batch_size = int(args.batch_size)
    verbose_freq = int(args.verbose_freq)
    save_freq = int(args.save_freq)
    z_cov = float(args.z_cov)
    pl_cov = float(args.pl_cov)
    interpolate_num = int(args.interpolate_num)
    ema_decay = float(args.ema_decay)

    style_size = int(args.style_size)
    z_size = int(args.z_size)
    texture_size = int(args.texture_size)
    gen_channel = int(args.gen_channel)
    gen_nonlocal_loc = int(args.gen_nonlocal_loc)
    gen_lr = float(args.gen_lr)
    mapping_lr = float(args.mapping_lr)
    gen_lazy = int(args.gen_lazy)
    ema_coef = (gen_lazy*ln2)/((img_size**2)*(img_level-1)*ln2)

    disc_first_channel = int(args.disc_first_channel)
    disc_last_size = int(args.disc_last_size)
    disc_nonlocal_loc = int(args.disc_nonlocal_loc)
    disc_lr = float(args.disc_lr)
    gp_coef = float(args.gp_coef)

    if device == 'multi':
        device = 'cuda:0'
        use_multi_gpu = True
    elif device == 'cpu':
        use_multi_gpu = False
    elif 'cuda:' in device:
        use_multi_gpu = False
    else:
        raise Exception('invalid argument in device (main)')

    #basic transformation for our dataset
    basic_transform = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)])
    dataset = TANOCIv2_Dataset(img_size=img_size, dataset_path=dataset_path, transform=basic_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    S = Networkv1.StyleMapper(z_size, style_size, mapping_lr, device)
    if version == 1:
        G = Networkv1.Generator(gen_channel, texture_size, style_size, gen_nonlocal_loc, gen_lr, img_size, device)
        D = Networkv1.Discriminator(disc_first_channel, disc_last_size, disc_nonlocal_loc, disc_lr, img_size, device)
    elif version == 2:
        G = Networkv2.Generator(gen_channel, texture_size, style_size, gen_nonlocal_loc, gen_lr, img_size, device)
        D = Networkv2.Discriminator(disc_first_channel, disc_last_size, disc_nonlocal_loc, disc_lr, img_size, device)
    else:
        raise Exception('invalid version')

    if torch.cuda.device_count() > 1 and use_multi_gpu:
        print('Using ', torch.cuda.device_count(), 'GPUs...')
        S = DataParallel(S)
        G = DataParallel(G)
        D = DataParallel(D)
    print('Parameter numbers: S, G, D')
    print(count_parameters(S), count_parameters(G), count_parameters(D))
    #visual seed(interpolation applied)
    dist = MultivariateNormal(loc=torch.zeros(batch_size//interpolate_num,z_size), covariance_matrix=z_cov*torch.eye(z_size))
    v = dist.sample().numpy()
    v_len = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v /= v_len
    v *= float(np.sqrt(z_size))
    visual_seed = torch.FloatTensor(np.linspace(v, -v, interpolate_num).transpose(1,0,2).reshape((-1,z_size))).to(device)
    #baseline
    if not lir:
        dist = MultivariateNormal(loc=torch.zeros(batch_size, z_size), covariance_matrix=z_cov*torch.eye(z_size))
    pl_dist = MultivariateNormal(loc=torch.zeros(batch_size, 3*img_size*img_size), covariance_matrix=pl_cov*torch.eye(3*img_size*img_size))
    previous_grads_norm = 0
    step_cnt = 1
    verbose_cnt = verbose_freq - 1
    for e in range(epoch):
        for real in dataloader:
            real = real.to(device)
            real_batch_size = real.size()[0]
            #baseline
            if not lir:
                z = torch.FloatTensor(dist.sample()).to(device)[:real_batch_size]
            else:
                #Linear interpolation regulation
                v = dist.sample().numpy()
                v_len = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
                v /= v_len
                v = v.reshape(batch_size//interpolate_num,z_size,1)
                epsilon = np.random.normal(0, 1, size=(batch_size//interpolate_num, interpolate_num)).reshape((batch_size//interpolate_num,1,interpolate_num))
                v = v*epsilon
                v = v.transpose((0,2,1)).reshape(batch_size, -1)
                z = torch.FloatTensor(v).to(device)[:real_batch_size]
            print('------------!--------------!------------')
            #d_update
            G_fake = G(S(z))
            fake = G_fake.detach()
            real_out = D(real)
            fake_out = D(fake)
            d_loss = -(torch.mean(real_out) - torch.mean(fake_out))
            print('epoch:', e, 'd_loss', d_loss)
            #gradient penalty for WGAN_GP
            epsilon = torch.rand(real_batch_size, 1, 1, 1).to(device)
            interpolated = epsilon*fake + (1-epsilon)*real
            interpolated.requires_grad_()
            interpolated_out = D(interpolated)
            grads = torch.autograd.grad(interpolated_out, interpolated, 
                grad_outputs=torch.ones_like(interpolated_out).to(device), 
                retain_graph=True, create_graph=True
            )[0]
            grads = grads.view(real_batch_size, -1)
            grad_penalty = torch.mean((grads.norm(2, dim=1)-1)**2)
            print('WGAN grad_penalty: ', float(grad_penalty.detach().cpu().numpy()))
            d_loss += gp_coef*grad_penalty
            if torch.cuda.device_count() > 1 and use_multi_gpu:
                D.module.opt.zero_grad()
            else:
                D.opt.zero_grad()      
            d_loss.backward()
            if torch.cuda.device_count() > 1 and use_multi_gpu:
                D.module.opt.step()
            else:
                D.opt.step() 
            #g_update
            fake_out = D(G_fake)
            g_loss = -torch.mean(fake_out)
            print('epoch:', e, '!!!g_loss', g_loss)
            if step_cnt % gen_lazy == 0 and version == 2:
                #path length Regulation
                y = pl_dist.sample().to(device)[:real_batch_size]
                w = S(z)
                Gw = G(w).view(real_batch_size, -1)
                Jy = torch.bmm(Gw.view(real_batch_size, 1, -1), y.view(real_batch_size, -1, 1))
                grads = torch.autograd.grad(Jy, w, 
                    grad_outputs=torch.ones_like(Jy).to(device),
                    retain_graph=True, create_graph=True
                )[0]
                grads = grads.view(real_batch_size, -1)
                grads_norm = grads.norm(2, dim=1)
                current_grads_norm = float(torch.mean(grads_norm).detach().cpu().numpy())
                grad_penalty = torch.mean(grads_norm - previous_grads_norm)**2
                print('gen_grad_penalty', float(grad_penalty.detach().cpu().numpy()))
                previous_grads_norm = previous_grads_norm*ema_decay + (1-ema_decay)*current_grads_norm
                print('a:', previous_grads_norm)
                g_loss += ema_coef*grad_penalty
            if torch.cuda.device_count() > 1 and use_multi_gpu:                        
                S.module.opt.zero_grad()
                G.module.opt.zero_grad()
            else:
                S.opt.zero_grad()
                G.opt.zero_grad()
            g_loss.backward()
            if torch.cuda.device_count() > 1 and use_multi_gpu:
                S.module.opt.step()
                G.module.opt.step()
            else:
                S.opt.step()
                G.opt.step()
            step_cnt += 1
            verbose_cnt += 1
            if verbose_cnt % verbose_freq == 0:
                img_save_path = save_path + '_img/'
                index = str(verbose_cnt//verbose_freq)
                vis_fake = G(S(visual_seed)).detach()
                save_image(make_grid(vis_fake), img_save_path + index + '.jpg')
                save_image(make_grid(vis_fake), args.lir + '.jpg')
                if int(index) % save_freq == 0:
                    weight_save_path = save_path + '_weight/'
                    torch.save(S.state_dict(), weight_save_path + index + 'S.pt')
                    torch.save(G.state_dict(), weight_save_path + index +  'G.pt')
                    torch.save(D.state_dict(), weight_save_path + index + 'D.pt')