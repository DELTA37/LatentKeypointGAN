import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
from .ops import fused_leaky_relu, FusedLeakyReLU, upfirdn2d


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
    y_coords = 2.0 * torch.arange(image_height, dtype=kp_pos.dtype, device=kp_pos.device).unsqueeze(1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
    x_coords = 2.0 * torch.arange(image_width, dtype=kp_pos.dtype, device=kp_pos.device).unsqueeze(0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
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
        kp_emb = torch.cat([kp_emb, bg_emb], dim=1)

        latent = torch.cat([kp_pos.view(-1, 2 * self.kps_num),
                            kp_emb.view(-1, (self.kps_num + 1) * self.noise_dim)], dim=1)
        return latent

        return [kp_pos,  # bs x 2 x kps_num
                kp_emb]  # bs x kps_num + 1 x 512


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
    def __init__(self,
                 size=512,
                 noise_dim=512,
                 channel_multiplier=4,
                 kps_num=5,
                 n_mlp=3,
                 lr_mlp=1e-2):
        super(SPADEGenerator, self).__init__()
        assert size in [256, 512]
        nf = channel_multiplier
        self.noise_dim = noise_dim
        self.kps_num = kps_num
        self.size = size

        self.input = ConstantInput(128 * nf)
        self.mapping = Mapping(kps_num=kps_num,
                               noise_dim=noise_dim,
                               n_mlp=n_mlp,
                               lr_mlp=lr_mlp)

        self.up = nn.Upsample(scale_factor=2)

        self.up_0 = SPADEBlock(128 * nf, 128 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_0 = nn.Conv2d(128 * nf, 3,
                                  kernel_size=1)

        self.up_1 = SPADEBlock(128 * nf, 128 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_1 = nn.Conv2d(128 * nf, 3,
                                  kernel_size=1)

        self.up_2 = SPADEBlock(128 * nf, 128 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_2 = nn.Conv2d(128 * nf, 3,
                                  kernel_size=1)

        self.up_3 = SPADEBlock(128 * nf, 64 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_3 = nn.Conv2d(64 * nf, 3,
                                  kernel_size=1)

        self.up_4 = SPADEBlock(64 * nf, 32 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_4 = nn.Conv2d(32 * nf, 3,
                                  kernel_size=1)

        self.up_5 = SPADEBlock(32 * nf, 16 * nf,
                               noise_dim=noise_dim)
        self.to_rgb_5 = nn.Conv2d(16 * nf, 3,
                                  kernel_size=1)

        if size >= 512:
            self.up_6 = SPADEBlock(16 * nf, 8 * nf,
                                   noise_dim=noise_dim)
            self.to_rgb_6 = nn.Conv2d(8 * nf, 3,
                                      kernel_size=1)

    def parse_latent(self, latent):
        kp_pos, kp_emb = torch.split(latent, (2 * self.kps_num, (self.kps_num + 1) * self.noise_dim), dim=1)
        kp_pos = kp_pos.view(-1, 2, self.kps_num)
        kp_emb = kp_emb.view(-1, (self.kps_num + 1), self.noise_dim)
        info = Munch(kp_pos=kp_pos,
                     kp_emb=kp_emb)
        return info

    def forward(self, z,
                return_latents=True):
        z_kp_pose, z_kp_emb, z_bg_emb = torch.split(z, (self.noise_dim, self.noise_dim, self.noise_dim), dim=1)
        latent = self.mapping(z_kp_pose, z_kp_emb, z_bg_emb)
        info = self.parse_latent(latent)
        x = self.input(z_kp_pose)  # 4

        x = self.up(x)  # 8
        x = self.up_0(x, info)  # 8
        y = self.to_rgb_0(x)
        x = self.up(x)  # 16
        x = self.up_1(x, info)  # 16
        y = self.up(y) + self.to_rgb_1(x)
        x = self.up(x)  # 32
        x = self.up_2(x, info)  # 32
        y = self.up(y) + self.to_rgb_2(x)
        x = self.up(x)  # 64
        x = self.up_3(x, info)  # 64
        y = self.up(y) + self.to_rgb_3(x)
        x = self.up(x)  # 128
        x = self.up_4(x, info)  # 128
        y = self.up(y) + self.to_rgb_4(x)
        x = self.up(x)  # 256
        x = self.up_5(x, info)  # 256
        y = self.up(y) + self.to_rgb_5(x)
        if self.size >= 512:
            x = self.up(x)  # 512
            x = self.up_6(x, info)  # 512
            y = self.up(y) + self.to_rgb_6(x)

        if return_latents:
            return y, info.kp_emb
        return y


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input: torch.Tensor):
        out = upfirdn2d(input, self.kernel, pad=self.pad, torch_impl=input.dtype == torch.float16)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 activation=None):
        super(Discriminator, self).__init__()

        channels = {
            4: 128,
            8: 128,
            16: 128,
            32: 128,
            64: 64 * channel_multiplier,
            128: 32 * channel_multiplier,
            256: 16 * channel_multiplier,
            512: 8 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        self.activation = activation

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        if self.activation is not None:
            out = self.activation(out)

        return out
