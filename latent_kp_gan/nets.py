import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
from .ops import fused_leaky_relu


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim,
                 bias_init=0,
                 lr_mul=1.,
                 activation=False):
        super(EqualLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation is not None:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale,
                           bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class KeypointGenerator(nn.Module):
    def __init__(self,
                 kps_num=5,
                 noise_dim=512,
                 n_mlp=3,
                 lr_mul=1.):
        super(KeypointGenerator, self).__init__()
        self.kps_num = kps_num
        self.noise_dim = noise_dim
        self.n_mlp = n_mlp

        self.layers = nn.Sequential(*([EqualLinear(noise_dim, noise_dim,
                                                   lr_mul=lr_mul,
                                                   activation=True)
                                       for _ in range(n_mlp - 1)] + [EqualLinear(noise_dim, 2 * kps_num,
                                                                                 lr_mul=lr_mul,
                                                                                 activation=False)]))

    def forward(self, z):
        return F.tanh(self.layers(z)).view(-1, 2, self.kps_num)


def kp2heatmap(kp_pos,
               image_height, image_width,
               sigma=1.):
    """

    :param kp_pos: bs x 2 x kps_num
    :return:
    """
    y_coords = 2.0 * torch.arange(image_height, dtype=kp_pos.dtype).unsqueeze(1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
    x_coords = 2.0 * torch.arange(image_width, dtype=kp_pos.dtype).unsqueeze(0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
    coords = torch.stack((y_coords, x_coords), dim=0)
    coords = torch.unsqueeze(coords, dim=0)  # 1 x 2 x h x w

    H = torch.exp(-torch.square(torch.unsqueeze(coords, dim=2) - kp_pos.unsqueeze(3).unsqueeze(3)).sum(dim=1) / sigma)  # bs x kps_num x h x w
    H = torch.cat([H, 1 - torch.max(H, dim=1, keepdim=True).values], dim=1)
    return H


class Mapping(nn.Module):
    def __init__(self,
                 kps_num=5,
                 noise_dim=512,
                 n_mlp=3,
                 lr_mlp=0.01):
        super(Mapping, self).__init__()

        self.kps_num = kps_num
        self.noise_dim = noise_dim
        self.n_mlp = n_mlp

        self.kp_pose = KeypointGenerator(kps_num=kps_num,
                                         noise_dim=noise_dim,
                                         n_mlp=n_mlp,
                                         lr_mul=lr_mlp)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(noise_dim,
                            noise_dim,
                            lr_mul=lr_mlp,
                            activation=True)
            )
        self.style = nn.Sequential(*layers)

        bg_layers = [PixelNorm()]
        for i in range(n_mlp):
            bg_layers.append(
                EqualLinear(noise_dim,
                            noise_dim,
                            lr_mul=lr_mlp,
                            activation=True)
            )
        self.bg_style = nn.Sequential(*bg_layers)

        self.kp_constant_emb = nn.Parameter(torch.randn(1, kps_num, noise_dim))

    def forward(self, z_kp_pose, z_kp_emb, z_bg_emb):
        kp_global_emb = self.style(z_kp_emb).view(-1, 1, self.noise_dim)  # bs x 512
        kp_emb = self.kp_constant_emb * kp_global_emb  # bs x kps_num x 512
        kp_pos = self.kp_pose(z_kp_pose)
        bg_emb = self.bg_style(z_bg_emb).view(-1, 1, self.noise_dim)
        return Munch(kp_pos=kp_pos,  # bs x 2 x kps_num
                     kp_emb=torch.cat([kp_emb, bg_emb], dim=1))  # bs x kps_num + 1 x 512


class SPADE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 ks=3,
                 kps_num=5,
                 noise_dim=512,
                 sigma=5.):
        super(SPADE, self).__init__()
        self.kps_num = kps_num
        self.noise_dim = noise_dim
        self.sigma = sigma

        self.param_free_norm = nn.BatchNorm2d(in_channels,
                                              affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(noise_dim * (kps_num + 1),
                      hidden_channels,
                      kernel_size=ks,
                      padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels,
                                   in_channels,
                                   kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(hidden_channels,
                                  in_channels,
                                  kernel_size=ks, padding=pw)

    def forward(self, x, info):
        normalized = self.param_free_norm(x)
        _, _, image_height, image_width = normalized.size()
        H = kp2heatmap(info.kp_pos,
                       image_height=image_height,
                       image_width=image_width,
                       sigma=self.sigma)  # bs x kps_num + 1 x h x w
        H = (H.unsqueeze(2) * info.kp_emb.unsqueeze(3).unsqueeze(3)).view(-1, (1 + self.kps_num) * self.noise_dim, image_height, image_width)

        actv = self.mlp_shared(H)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEBlock(nn.Module):
    def __init__(self,
                 fin,
                 fout,
                 ks=3,
                 kps_num=5,
                 noise_dim=512):

        super(SPADEBlock, self).__init__()
        fmiddle = min(fin, fout)
        hidden_channels = fmiddle

        self.conv_0 = nn.Conv2d(fin, fmiddle,
                                kernel_size=ks, padding=ks // 2)
        self.conv_1 = nn.Conv2d(fmiddle, fout,
                                kernel_size=ks, padding=ks // 2)

        self.norm_0 = SPADE(fin,
                            hidden_channels,
                            ks=ks,
                            kps_num=kps_num,
                            noise_dim=noise_dim)

        self.norm_1 = SPADE(fmiddle,
                            hidden_channels,
                            ks=ks,
                            kps_num=kps_num,
                            noise_dim=noise_dim)

    def forward(self, x, info):
        x = self.conv_0(self.actvn(self.norm_0(x, info)))
        x = self.conv_1(self.actvn(self.norm_1(x, info)))
        return x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class SPADEGenerator(nn.Module):
    def __init__(self, kps_num=5,
                 noise_dim=512,
                 n_mlp=3,
                 lr_mlp=1e-2,
                 nf=3):
        super(SPADEGenerator, self).__init__()
        self.noise_dim = noise_dim

        self.input = ConstantInput(128 * nf)
        self.mapping = Mapping(kps_num=kps_num,
                               noise_dim=noise_dim,
                               n_mlp=n_mlp,
                               lr_mlp=lr_mlp)

        self.up = nn.Upsample(scale_factor=2)

        self.up_0 = SPADEBlock(128 * nf, 64 * nf,
                               noise_dim=noise_dim)
        self.up_1 = SPADEBlock(64 * nf, 32 * nf,
                               noise_dim=noise_dim)
        self.up_2 = SPADEBlock(32 * nf, 16 * nf,
                               noise_dim=noise_dim)
        self.up_3 = SPADEBlock(16 * nf, 8 * nf,
                               noise_dim=noise_dim)
        self.up_4 = SPADEBlock(8 * nf, 4 * nf,
                               noise_dim=noise_dim)
        self.up_5 = SPADEBlock(4 * nf, 2 * nf,
                               noise_dim=noise_dim)
        self.up_6 = SPADEBlock(2 * nf, 1 * nf,
                               noise_dim=noise_dim)

    def forward(self, z):
        z_kp_pose, z_kp_emb, z_bg_emb = torch.split(z, (self.noise_dim, self.noise_dim, self.noise_dim), dim=1)
        info = self.mapping(z_kp_pose, z_kp_emb, z_bg_emb)
        x = self.input(z_kp_pose)  # 4

        x = self.up(x)  # 8
        x = self.up_0(x, info)  # 8
        x = self.up(x)  # 16
        x = self.up_1(x, info)  # 16
        x = self.up(x)  # 32
        x = self.up_2(x, info)  # 32
        x = self.up(x)  # 64
        x = self.up_3(x, info)  # 64
        x = self.up(x)  # 128
        x = self.up_4(x, info)  # 128
        x = self.up(x)  # 256
        x = self.up_5(x, info)  # 256
        x = self.up(x)  # 512
        x = self.up_6(x, info)  # 512
        return x
