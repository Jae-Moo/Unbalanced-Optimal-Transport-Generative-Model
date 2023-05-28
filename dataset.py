# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 2. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# ---------------------------------------------------------------

import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from PIL import Image
import os.path


# Image datasets
class CelebA_HQ(data.Dataset):
    '''Note: CelebA (about 200000 images) vs CelebA-HQ (30000 images)'''
    def __init__(self, root, partition_path, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        # Split train/val/test 
        self.partition_dict = {}
        self.get_partition_label(partition_path)
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.save_img_path()
        print('[Celeba-HQ Dataset]')
        print(f'Train {len(self.train_dataset)} | Val {len(self.val_dataset)} | Test {len(self.test_dataset)}')

        if mode == 'train':
            self.dataset = self.train_dataset
        elif mode == 'val':
            self.dataset = self.val_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError

    def get_partition_label(self, list_eval_partition_celeba_path):
        '''Get partition labels (Train 0, Valid 1, Test 2) from CelebA
        See "celeba/Eval/list_eval_partition.txt"
        '''
        with open(list_eval_partition_celeba_path, 'r') as f:
            for line in f.readlines():
                filenum = line.split(' ')[0].split('.')[0] # Use 6-digit 'str' instead of int type
                partition_label = int(line.split(' ')[1]) # 0 (train), 1 (val), 2 (test)
                self.partition_dict[filenum] = partition_label

    def save_img_path(self):
        for filename in os.listdir(self.root):
            assert os.path.isfile(os.path.join(self.root, filename))
            filenum = filename.split('.')[0]
            label = self.partition_dict[filenum]
            if label == 0:
                self.train_dataset.append(os.path.join(self.root, filename))
            elif label == 1:
                self.val_dataset.append(os.path.join(self.root, filename))
            elif label == 2:
                self.test_dataset.append(os.path.join(self.root, filename))
            else:
                raise ValueError

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


class AnomalyDataset(data.Dataset):
    def __init__(self, dataset, anomaly_dataset, frac=0.01):
        '''
        dataset : target dataset (CIFAR10)
        anomaly_dataset : anomaly dataset (MNIST)
        frac : fraction of anomaly dataset (p=0.01)
        '''
        try: normal_sample, _ = dataset[0]
        except: normal_sample = dataset[0]
        c, size, _ = normal_sample.shape # [c, w, h]
        
        self.dataset = dataset
        self.anomaly_dataset = anomaly_dataset

        self.num_normal = dataset.__len__()
        self.num_anomaly = int(frac * self.num_normal)
        
        self.ANOMALIES = []
        for i in range(self.num_anomaly):
            # get samples
            x = anomaly_dataset[i]
            try: x, _ = x
            except: pass
            # check if image size is same
            if i==0: assert x.shape[1] == size
            # match the number of channels
            if x.shape[0]==1 and c==3:
                x = x.repeat(3,1,1)
            # append to self.ANOMALIES
            self.ANOMALIES.append(x)
    
    def __getitem__(self, index):
        if index < self.num_normal:
            x = self.dataset[index]
            try: x, _ = x
            except: pass
        else:
            x = self.ANOMALIES[index-self.num_normal]
        
        return x

    def __len__(self):
        return self.num_normal + self.num_anomaly


# get dataloader
def get_dataloader(args):
    num_workers = 4
    if args.dataset == 'mnist':
        dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
    
    elif args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif args.dataset == 'cifar10+mnist':
        normal_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        
        anomaly_dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
        
        dataset = AnomalyDataset(normal_dataset, anomaly_dataset)
    
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root='data/celeba-hq/celeba-256',
            partition_path='data/celeba-hq/list_eval_partition_celeba.txt',
            mode='train', # 'train', 'val', 'test'
            transform=train_transform,
        )
    
    else: NotImplementedError
        
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader




# ------------------------
# For Toy
# ------------------------
# datasets
class ToydatasetGaussian(data.Dataset):
    def __init__(self, cfg):
        self.dataset = torch.randn(cfg.num_data, cfg.data_dim) + torch.tensor([0,10])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetp(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(cfg.num_data//2, cfg.data_dim)+1, 
                                  std*torch.randn(cfg.num_data-cfg.num_data//2, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Toydatasetq(data.Dataset):
    def __init__(self, cfg):
        std = 0.5
        self.dataset = torch.cat([std*torch.randn(2*cfg.num_data//3, cfg.data_dim)+2, 
                                  std*torch.randn(cfg.num_data-2*cfg.num_data//3, cfg.data_dim)-1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ToydatasetOutlier(data.Dataset):
    def __init__(self, cfg):
        M = int(cfg.num_data*cfg.p)
        self.dataset = torch.cat([0.1*torch.randn(cfg.num_data-M, cfg.data_dim) + 1, 0.1*torch.randn(M, cfg.data_dim) - 1])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ToydatasetNoise(data.Dataset):
    def __init__(self, cfg):
        self.N = cfg.num_data
        self.dim = cfg.data_dim
    
    def __len__(self):
        return int(self.N)
        
    
    def __getitem__(self, idx):
        return torch.randn((1, self.dim))


def get_datasets(cfg):
    src_name, tar_name = cfg.source_name, cfg.target_name
    datasets = []

    for name in [src_name, tar_name]:
        if name == 'gaussian':
            dataset = ToydatasetGaussian(cfg)
        elif name == 'p':
            dataset = Toydatasetp(cfg)
        elif name == 'q':
            dataset = Toydatasetq(cfg)
        elif name == 'outlier':
            dataset = ToydatasetOutlier(cfg)
        elif name == 'noise':
            dataset = ToydatasetNoise(cfg)
        else:
            raise NotImplementedError
        
        datasets.append(dataset)
    
    return datasets
