# Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport
<p align="middle">
  <img src="assets/cifar10_main.png" width="49%" />
  <img src="assets/celeba_main.png" width="49%" /> 
</p>
We propose a novel generative model using the semi-dual formulation of Unbalanced Optimal Transport (UOT). This approach provides robustness against outliers, stability during training, and fast convergence. Our algorithm is simple, but performs extremely well, achieving FID scores of 2.97 on CIFAR-10 and 5.80 on CelebA-HQ-256.
Precisely, the semi-dual form of the UOT problem can be reduced to the following objective:

$$\inf_{v_\phi}\left[ \int_{\mathcal{X}} \Psi_1^* \left( -\inf_{T_\theta} \left[c\left(x,T_\theta(x)\right)-v\left(T_\theta(x)\right)\right] \right) d\mu(x) + \int_{\mathcal{Y}} \Psi^*_2\left(-v(y)\right) d\nu(y) \right],$$

where $v_\phi$ is a discrimiator (potential), and $T_\theta$ is a generator (OT-map).
Here, $\Psi^*$ should be a non-decreasing, differentiable, convex function.

## Training UOTM ##
We use the following commands for training UOTM.
To train with other $\Psi^*$, just simply adjust "phi1" and "phi2" arguments.
If you want to use $\Psi^*$ other than implemented ones, add your function to the ``select_phi`` in ``utils.py``.

#### Toy ####
Commands for toy experiments are as follows.

Outlier Experiment
```
python train_toy.py --exp outlier \
    --num_data 4000 \
    --data_dim 1 \
    --p 0.01 \
    --source_name noise \
    --target_name outlier \
    --phi1 kl \
    --phi2 kl \
    --tau 0.1 \
    --regularize \
    --lmbda 0.01 \
    --epochs 400 \
    --batch_size 256 --savepath train_logs/outlier/outlier_uot --lr 1e-4
```
OT map Experiment
```
python train_toy.py --exp 1d-gaussian-mixture \
    --num_data 4000 \
    --data_dim 1 \
    --source_name p \
    --target_name q \
    --phi1 kl \
    --phi2 kl \
    --tau 0.02 \
    --epochs 2000 \
    --batch_size 256 --savepath train_logs/GM/GM_uot
```
#### Outlier Experiment ####
We train image outlier experiments using 4 32-GB V100 GPU.
```
python train.py --dataset cifar10+mnist --exp kl_anomaly --phi1 kl --phi2 kl
```

#### CIFAR-10 ####
We train UOTM on CIFAR-10 using 4 32-GB V100 GPU. 
```
python train.py --exp kl --phi1 kl --phi2 kl
```

#### CelebA-HQ-256 ####
We train UOTM on CelebA-HQ-256 using 8 32-GB V100 GPU. 
```
python train.py --dataset celeba_256 \\
                --exp kl --phi1 kl --phi2 kl --image_size 256 \\
                --num_channels_dae 64 --n_mlp 3 --ch_mult 1 1 2 2 4 4 \\
                --lr_d 1e-4 --lr_g 2e-4 --schedule 700 --ema_decay 0.999 \\
                --batch_size 32 --num_epoch 450 \\
                --tau 0.00001 --r1_gamma 5
```


## Pretrained Checkpoints ##
Pretrained checkpoints on CIFAR-10 and CelebA HQ 256 will be available soon. 


## Bibtex ##
Cite our paper using the following BibTeX item:
```
@misc{choi2023generative,
      title={Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport}, 
      author={Jaemoo Choi and Jaewoong Choi and Myungjoo Kang},
      year={2023},
      eprint={2305.14777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
