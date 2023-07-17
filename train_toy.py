import numpy as np
import torch
from models.toy import ToyDiscriminator, ToyGenerator
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import grad
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from utils import get_datasets, select_phi


def main(cfg):

    # dataloader
    src_dataset, tar_dataset = get_datasets(cfg)
    src_dataloader = DataLoader(src_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    tar_dataloader = DataLoader(tar_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # model
    netG = ToyGenerator(data_dim=cfg.data_dim, z_dim=cfg.data_dim).cuda()
    netD = ToyDiscriminator(data_dim=cfg.data_dim).cuda()

    # optimizer
    optimizerD = Adam(netD.parameters(), lr=cfg.lr)
    optimizerG = Adam(netG.parameters(), lr=cfg.lr)

    # phi
    phi1 = select_phi(cfg.phi1)
    phi2 = select_phi(cfg.phi2)

    # make savepath
    os.makedirs(cfg.savepath, exist_ok=True)


    # train
    tau, lmbda = cfg.tau, cfg.lmbda
    regularize = cfg.regularize
    if not regularize: lmbda = 0

    for epoch in range(cfg.epochs):
        for x_src, x_tar in zip(src_dataloader, tar_dataloader):
            x_src = x_src.float().cuda()
            x_tar = x_tar.float().cuda()

            # update D
            for _ in range(5):
                optimizerD.zero_grad()
                z = torch.randn_like(x_src)

                # with torch.no_grad():
                x_src.requires_grad = True
                x_pred = netG(x_src, z)
                logit = netD(x_pred)
                errD = phi1(logit - 0.5*tau*torch.sum((x_pred-x_src)**2, dim=1)) + phi2(-netD(x_tar))
                
                if regularize:
                    reg = grad(logit.sum(), x_src, create_graph=True)[0]
                    errD = errD + lmbda * (reg.norm(2, dim=1)**2).mean()

                errD.mean().backward()
                optimizerD.step()

            # update G
            optimizerG.zero_grad()

            z = torch.randn_like(x_src)
            x_pred = netG(x_src, z)
            
            errG = 0.5*tau*torch.sum((x_pred-x_src)**2, dim=1) - netD(x_pred)
            errG.mean().backward()
            optimizerG.step()
        

        # evaluation
        if epoch % cfg.save_every == cfg.save_every - 1:
            # evaluation for 1D tasks
            if cfg.data_dim == 1:
                with torch.no_grad():
                    sources = []
                    preds = []

                    try: x_src = src_dataset.dataset.to(x_tar.device)
                    except: x_src = torch.randn((cfg.num_data, 1), device=x_tar.device)
                    # x_src = torch.randn((cfg.num_data, 1), device=x_tar.device)
                    
                    z = torch.randn_like(x_src)
                    x_pred = netG(x_src, z)
                    sources.append(x_src.detach().cpu().numpy())
                    preds.append(x_pred.detach().cpu().numpy())

                    sources = np.concatenate(sources)
                    preds = np.concatenate(preds)

                    # joint distribution scatter plot
                    plt.scatter(sources[:,0], preds[:,0])
                    plt.xlabel('source')
                    plt.ylabel('target')
                    plt.savefig(os.path.join(cfg.savepath, 
                                             f'joint_{cfg.exp}_{cfg.phi1}_{cfg.phi2}_num{cfg.num_data}_out{cfg.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.png'))
                    plt.close()

                    # target density plot
                    df = pd.DataFrame({'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})
                    np.save(os.path.join(cfg.savepath, f'{cfg.exp}_{cfg.phi1}_{cfg.phi2}_num{cfg.num_data}_out{cfg.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.npy'), 
                                         {'source_density': x_src.cpu().numpy()[:,0], 'target density': tar_dataset.dataset.detach().cpu().numpy()[:,0], 'generated density': preds[:,0]})
                    sns_plot = sns.kdeplot(df, fill=True, y=None)
                    fig = sns_plot.get_figure()
                    # plt.xlim(-2,2)
                    # plt.ylim(0,2.5)
                    plt.xticks([])
                    plt.yticks([]) 
                    ax = plt.gca()
                    ax.get_yaxis().set_visible(False)
                    plt.tight_layout()
                    fig.savefig(os.path.join(cfg.savepath, f"target_{cfg.exp}_{cfg.phi1}_{cfg.phi2}_num{cfg.num_data}_out{cfg.p}_tau{tau}_lmbda{lmbda}_epoch{epoch+1}.png"))
                    
                    plt.close()
            
            if cfg.data_dim == 2:
                pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser('toy parameters')
    
    # Experiment description
    parser.add_argument('--exp', type=str, required=True, choices=['gaussian', 'outlier', '1d-gaussian-mixture'], help='experiment name')
    parser.add_argument('--num_data', type=int, default=4000, help='Number of data points')
    parser.add_argument('--data_dim', type=int, default=1, help='The dimensiion of data')
    parser.add_argument('--source_name', type=str, required=True, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of source dataset')
    parser.add_argument('--target_name', type=str, required=True, choices=['gaussian', 'p', 'q', 'outlier', 'noise'], help='Name of target dataset')    
    parser.add_argument('--p', type=float, default=0., help='Only for outlier test. Fraction of outlier')
    
    # settings
    parser.add_argument('--phi1', type=str, default='linear', choices=['linear', 'kl'], help='Choices of phi1 star')
    parser.add_argument('--phi2', type=str, default='linear', choices=['linear', 'kl'], help='Choices of phi2 star')
    parser.add_argument('--tau', type=float, required=True, help='scalar value multiplied to quadratic cost functional')
    parser.add_argument('--regularize', action='store_true', default=False, help='use regularization or not')
    parser.add_argument('--lmbda', type=float, default=0.01, help='regularization hyperparameter')

    # training configurations
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')

    # kl-div
    parser.add_argument('--kl_estimator', type=str, default='scipy', help='kl divergence estimator')

    # save path
    parser.add_argument('--savepath', type=str, required=True, help='experiment save path')
    parser.add_argument('--save_every', type=int, default=100, help='Evaluation every {save_every} epoch')

    args = parser.parse_args()
    main(args)
