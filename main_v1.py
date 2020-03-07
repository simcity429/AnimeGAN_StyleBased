import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from config import *
from Networkv1 import Discriminator, StyleMapper, Generator
from CustomDataset import TANOCIv2_Dataset

if __name__ == '__main__':
    dataset = TANOCIv2_Dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    S = StyleMapper()
    G = Generator()
    D = Discriminator()
    #Normal
    
    dist = MultivariateNormal(loc=torch.zeros(BATCH_SIZE, Z_SIZE), covariance_matrix=COV*torch.eye(Z_SIZE))
    visual_seed = torch.FloatTensor(dist.sample()).to(DEVICE)
    
    #Linear interpolation regulation
    '''
    dist = MultivariateNormal(loc=torch.zeros(BATCH_SIZE//INTERPOLATE_NUM,Z_SIZE), covariance_matrix=COV*torch.eye(Z_SIZE))
    tmp = dist.sample().numpy()
    visual_seed = torch.FloatTensor(np.linspace(tmp, -tmp, INTERPOLATE_NUM).transpose(1,0,2).reshape((-1,Z_SIZE))).to(DEVICE)
    '''
    verbose_cnt = VERBOSE_CNT - 1
    for e in range(EPOCH):
        for real in dataloader:
            real = real.to(DEVICE)
            batch_size = real.size()[0]
            #Normal
            z = torch.FloatTensor(dist.sample()).to(DEVICE)[:batch_size]
            #Linear interpolation regulation
            '''
            tmp = dist.sample().numpy()
            z = torch.FloatTensor(np.linspace(tmp, -tmp, INTERPOLATE_NUM).transpose(1,0,2).reshape((-1,Z_SIZE))).to(DEVICE)[:batch_size]
            '''
            print('------------!--------------!------------')
            #d_update
            G_fake = G(S(z))
            fake = G_fake.detach()
            real_out = D(real)
            fake_out = D(fake)
            d_loss = -(torch.mean(real_out) - torch.mean(fake_out))
            print('epoch:', e, 'd_loss', d_loss)
            #gradient penalty
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
            reg = torch.mean(interpolated_out)**2
            print('grad_penalty: ', grad_penalty)
            d_loss += GP_COEF*grad_penalty + REG*reg
            #update parameter
            D.opt.zero_grad()
            d_loss.backward()
            D.opt.step()
            #g_update
            fake_out = D(G_fake)
            g_loss = -torch.mean(fake_out)
            print('epoch:', e, '!g_loss', g_loss)
            S.opt.zero_grad()
            G.opt.zero_grad()
            g_loss.backward()
            S.opt.step()
            G.opt.step()
            verbose_cnt += 1
            if verbose_cnt % VERBOSE_CNT == 0:
                index = str(verbose_cnt//VERBOSE_CNT)
                vis_fake = G(S(visual_seed)).detach()
                save_image(make_grid(vis_fake), IMG_SAVE_PATH + index + '.jpg')
                save_image(make_grid(vis_fake), './tmp.jpg')
                if int(index) % SAVE_FREQ == 0:
                    torch.save(S.state_dict(), WEIGHT_SAVE_PATH + index + 'S.pt')
                    torch.save(G.state_dict(), WEIGHT_SAVE_PATH + index +  'G.pt')
                    torch.save(D.state_dict(), WEIGHT_SAVE_PATH + index + 'D.pt')