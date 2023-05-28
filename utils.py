# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 2. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# 3. https://github.com/nhartland/KL-divergence-estimators (MIT License)
# ---------------------------------------------------------------

import warnings
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from scipy.spatial import KDTree


# ------------------------
# Select Phi_star
# ------------------------
def select_phi(name):
    if name == 'linear':
        def phi(x):
            return x
            
    elif name == 'kl':
        def phi(x):
            return torch.exp(x)
    
    elif name == 'chi':
        def phi(x):
            y = F.relu(x+2)-2
            return 0.25 * y**2 + y
        
    elif name == 'softplus':
        def phi(x):
            return F.softplus(x)
    else:
        raise NotImplementedError
    
    return phi

# ------------------------
# EMA
# ------------------------
class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        '''
        EMA Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
        '''
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()


# ------------------------
# Get Model
# ------------------------
# Get pretrained model
def get_model(args, ckpt_path):
    from models.ncsnpp_generator_adagn import NCSNpp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = NCSNpp(args).to(device)
    netG = torch.nn.DataParallel(netG, device_ids=[0,1,2,3])
    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint)
    for p in netG.parameters():
        p.requires_grad = False
    return netG



# ------------------------
# KL divergence
# ------------------------
def knn_distance(point, sample, k):
    """Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample`

    This function works for points in arbitrary dimensional spaces.
    """
    # Compute all euclidean distances
    norms = np.linalg.norm(sample - point, axis=1)
    # Return the k-th nearest
    return np.sort(norms)[k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]


def scipy_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scipy's KDTree
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d, nu_i = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k + 1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
    else:
        D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))

    return D

