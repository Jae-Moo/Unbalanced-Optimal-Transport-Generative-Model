import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from datetime import datetime
from utils import *
from dataset import get_dataloader




def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:0')
    batch_size = args.batch_size
    nz = args.nz
    
    
    # Set Generator
    from models.ncsnpp_generator_adagn import NCSNpp
    netG = NCSNpp(args).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.schedule, eta_min=1e-5)
    netG = nn.DataParallel(netG)
    
    
    # Set potential
    if args.dataset in ['mnist','cifar10','cifar10+mnist']:
        from models.discriminator import Discriminator_small
        netD = Discriminator_small(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2)).to(device)
    else:
        from models.discriminator import Discriminator_large
        netD = Discriminator_large(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2)).to(device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.schedule, eta_min=1e-5)
    netD = nn.DataParallel(netD)


    # Create log path
    exp = args.exp
    parent_dir = "./train_logs/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    os.makedirs(exp_path, exist_ok=True)
    
    # Get Data
    data_loader = get_dataloader(args)
    
    
    # Resume
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    
    # Make log file
    with open(os.path.join(exp_path, 'log.txt'), 'w') as f:
        f.write("Start Training")
        f.write('\n')
    
    
    # get phi star
    phi_star1 = select_phi(args.phi1)
    phi_star2 = select_phi(args.phi2)


    # Start training
    start = datetime.now()
    for epoch in range(init_epoch, args.num_epoch+1):
        for _, x in enumerate(data_loader):
            try: x,_ = x
            except: pass
            
            #### Update potential ####
            for p in netD.parameters():  
                p.requires_grad = True

            real_data = x.float().to(device, non_blocking=True)
            real_data.requires_grad = True
                
            netD.zero_grad()

            # real D loss
            noise = torch.randn_like(real_data)            
            D_real = netD(real_data)
            errD_real = phi_star2(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)
            
            # R1 regularization
            grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=real_data, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

            # fake D loss
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(noise, latent_z)
            D_fake = netD(x_0_predict)
            
            errD_fake = phi_star1(D_fake - 0.5 * args.tau * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1))
            errD_fake = errD_fake.mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()


            #### Update Generator ####
            for p in netD.parameters():
                p.requires_grad = False
            
            
            netG.zero_grad()

            # Generator loss
            noise = torch.randn_like(real_data)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(noise, latent_z)
            D_fake = netD(x_0_predict)
            
            err = args.tau * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1) - D_fake
            err = err.mean()
            err.backward()
            optimizerG.step()
            global_step += 1
            
            ## save losses
            if global_step % args.print_every == 0:
                with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                    f.write(f'Epoch {epoch:04d} : G Loss {err.item():.4f}, D Loss {errD.item():.4f}, Elapsed {datetime.now() - start}')
                    f.write('\n')
        

        schedulerG.step()
        schedulerD.step()
        
        # save content
        if epoch % args.save_content_every == 0:
            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                        'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                        'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            
            torch.save(content, os.path.join(exp_path, 'content.pth'))
        
        # save checkpoint
        if epoch % args.save_ckpt_every == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            
            torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_{}.pth'.format(epoch)))

        # save generated images
        if epoch % args.save_image_every == 0:
            noise = torch.randn_like(real_data)
            latent_z = torch.randn(batch_size, nz, device=device)
            images = netG(noise, latent_z)
            images = (0.5*(images+1)).detach().cpu()
            torchvision.utils.save_image(images, os.path.join(exp_path, 'epoch_{}.png'.format(epoch)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('UOT parameters')
    
    # Experiment description
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--exp', default='linear', help='name of experiment')
    parser.add_argument('--resume', action='store_true',default=False, help='Resume training or not')
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'cifar10+mnist', 'lsun', 'celeba_256'], help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
    # Generator configurations
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
    parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
    parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
    
    # Training/Optimizer configurations
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=600, help='the number of epochs')
    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.0e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--schedule', type=int, default=1800, help='cosine scheduler, learning rate 1e-5 until {schedule} epoch')
    parser.add_argument('--use_ema', action='store_false', default=True, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    # Loss configurations
    parser.add_argument('--phi1', type=str, default='kl', choices=['linear', 'kl', 'softplus'])
    parser.add_argument('--phi2', type=str, default='kl', choices=['linear', 'kl', 'softplus'])
    parser.add_argument('--tau', type=float, default=0.001, help='proportion of the cost c')
    parser.add_argument('--r1_gamma', type=float, default=0.2, help='coef for r1 reg')
        
    # Visualize/Save configurations
    parser.add_argument('--print_every', type=int, default=100, help='print current loss for every x iterations')
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=100, help='save ckpt every x epochs')
    parser.add_argument('--save_image_every', type=int, default=10, help='save images every x epochs')
    
    args = parser.parse_args()
    train(args)
