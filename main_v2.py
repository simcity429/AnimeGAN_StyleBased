import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.nn import BCEWithLogitsLoss

#WGAN-GP with path length reg
if __name__ == '__main__':
    import config
    parser = argparse.ArgumentParser()
    parser.add_argument('--lir', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--image_level', required=True)
    args = parser.parse_args()
    config.DEVICE = args.device
    config.SAVE_PATH = args.save_path
    config.BATCH_SIZE = int(args.batch_size)
    config.IMG_LEVEL = int(args.image_level)
    if args.lir == "True" or args.lir == "true":
        lir = True
    elif args.lir == "False" or args.lir == "false":
        lir = False
    else:
        raise NotImplementedError('wrong value on lir')
    from config import *
    from Networkv1 import StyleMapper
    from Networkv2 import Discriminator, Generator
    from CustomDataset import TANOCIv2_Dataset
    dataset = TANOCIv2_Dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    S = StyleMapper()
    G = Generator()
    D = Discriminator()
    #Linear interpolation regulation

    dist = MultivariateNormal(loc=torch.zeros(BATCH_SIZE//INTERPOLATE_NUM,Z_SIZE), covariance_matrix=COV*torch.eye(Z_SIZE))
    v = dist.sample().numpy()
    v_len = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v /= v_len
    v *= float(2*np.sqrt(Z_SIZE))
    visual_seed = torch.FloatTensor(np.linspace(v, -v, INTERPOLATE_NUM).transpose(1,0,2).reshape((-1,Z_SIZE))).to(DEVICE)
    
    #baseline
    if not lir:
        dist = MultivariateNormal(loc=torch.zeros(BATCH_SIZE, Z_SIZE), covariance_matrix=COV*torch.eye(Z_SIZE))
    
    pl_dist = MultivariateNormal(loc=torch.zeros(BATCH_SIZE, 3*IMG_SIZE*IMG_SIZE), covariance_matrix=PL_COV*torch.eye(3*IMG_SIZE*IMG_SIZE))
    previous_grads_norm = 0
    step_cnt = 1
    verbose_cnt = VERBOSE_CNT - 1
    for e in range(EPOCH):
        for real in dataloader:
            real = real.to(DEVICE)
            batch_size = real.size()[0]
            #baseline
            if not lir:
                z = torch.FloatTensor(dist.sample()).to(DEVICE)[:batch_size]
            else:
                #Linear interpolation regulation
                v = dist.sample().numpy()
                v_len = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
                v /= v_len
                v *= float(2*np.sqrt(Z_SIZE))
                v = v.reshape(BATCH_SIZE//INTERPOLATE_NUM,Z_SIZE,1)
                epsilon = np.random.uniform(low=0.0,high=1.0,size=INTERPOLATE_NUM).reshape((1,1,-1))
                v = v - 2*v*epsilon
                v = v.transpose((0,2,1)).reshape(BATCH_SIZE, -1)
                z = torch.FloatTensor(v).to(DEVICE)[:batch_size]

            print('------------!--------------!------------')
            #d_update
            G_fake = G(S(z))
            fake = G_fake.detach()
            real_out = D(real)
            fake_out = D(fake)
            d_loss = -(torch.mean(real_out) - torch.mean(fake_out))
            print('epoch:', e, 'd_loss', d_loss)
            epsilon = torch.rand(batch_size, 1, 1, 1).to(DEVICE)
            interpolated = epsilon*fake + (1-epsilon)*real
            interpolated.requires_grad_()
            interpolated_out = D(interpolated)
            grads = torch.autograd.grad(interpolated_out, interpolated, 
                grad_outputs=torch.ones_like(interpolated_out).to(DEVICE), 
                retain_graph=True, create_graph=True
            )[0]
            grads = grads.view(batch_size, -1)
            grad_penalty = torch.mean((grads.norm(2, dim=1)-1)**2)
            print('grad_penalty: ', float(grad_penalty.detach().cpu().numpy()))
            d_loss += GP_COEF*grad_penalty      
            D.opt.zero_grad()
            d_loss.backward()
            D.opt.step()
            #g_update
            fake_out = D(G_fake)
            g_loss = -torch.mean(fake_out)
            print('epoch:', e, '!g_loss', g_loss)
                        
            if step_cnt % GEN_LAZY_REG == 0:
                #pl Reg
                y = pl_dist.sample().to(DEVICE)[:batch_size]
                w = S(z)
                Gw = G(w).view(batch_size, -1)
                Jy = torch.bmm(Gw.view(batch_size, 1, -1), y.view(batch_size, -1, 1))
                grads = torch.autograd.grad(Jy, w, 
                    grad_outputs=torch.ones_like(Jy).to(DEVICE),
                    retain_graph=True, create_graph=True
                )[0]
                grads = grads.view(batch_size, -1)
                grads_norm = grads.norm(2, dim=1)
                current_grads_norm = float(torch.mean(grads_norm).detach().cpu().numpy())
                grad_penalty = torch.mean(grads_norm - previous_grads_norm)**2
                print('gen_grad_penalty', float(grad_penalty.detach().cpu().numpy()))
                previous_grads_norm = previous_grads_norm*EMA_DECAY + (1-EMA_DECAY)*current_grads_norm
                print('a:', previous_grads_norm)
                g_loss += EMA_COEF*grad_penalty
                        
            S.opt.zero_grad()
            G.opt.zero_grad()
            g_loss.backward()
            S.opt.step()
            G.opt.step()
            step_cnt += 1
            verbose_cnt += 1
            if verbose_cnt % VERBOSE_CNT == 0:
                img_save_path = SAVE_PATH + '_img/'
                index = str(verbose_cnt//VERBOSE_CNT)
                vis_fake = G(S(visual_seed)).detach()
                save_image(make_grid(vis_fake), img_save_path + index + '.jpg')
                save_image(make_grid(vis_fake), args.lir + '.jpg')
                if int(index) % SAVE_FREQ == 0:
                    weight_save_path = SAVE_PATH + '_weight/'
                    torch.save(S.state_dict(), weight_save_path + index + 'S.pt')
                    torch.save(G.state_dict(), weight_save_path + index +  'G.pt')
                    torch.save(D.state_dict(), weight_save_path + index + 'D.pt')