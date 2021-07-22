import argparse
import torch
from munch import Munch
import matplotlib.pyplot as plt
from latent_kp_gan.nets import SPADEGenerator, Mapping, kp2heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--noise_dim",
                        default=64,
                        type=int)
    parser.add_argument("--kps_num",
                        type=int,
                        default=5)
    parser.add_argument("--sigma",
                        default=0.1,
                        type=float)
    parser.add_argument("--size",
                        default=256,
                        type=int)
    args = parser.parse_args()
    return args


@torch.no_grad()
def test_heatmaps(batch_size=16,
                  kps_num=5,
                  noise_dim=64,
                  sigma=0.1,
                  size=256):
    mapping = Mapping(kps_num=kps_num,
                      noise_dim=noise_dim,
                      n_mlp=3,
                      lr_mlp=1e-2)
    z = torch.randn(batch_size, 3 * noise_dim)
    z_kp_pose, z_kp_emb, z_bg_emb = torch.split(z, (noise_dim, noise_dim, noise_dim), dim=1)
    latent = mapping(z_kp_pose, z_kp_emb, z_bg_emb)
    kp_pos, kp_emb = torch.split(latent, (2 * kps_num, (kps_num + 1) * noise_dim), dim=1)
    kp_pos = kp_pos.view(-1, 2, kps_num)
    kp_emb = kp_emb.view(-1, (kps_num + 1), noise_dim)
    info = Munch(kp_pos=kp_pos,
                 kp_emb=kp_emb)

    print(info.kp_pos.shape)
    H = kp2heatmap(info.kp_pos,
                   image_height=size,
                   image_width=size,
                   sigma=sigma)
    print(H.shape)
    for i in range(kps_num + 1):
        plt.imshow(H[0, i].numpy())
        plt.show()


if __name__ == "__main__":
    test_heatmaps(**vars(parse_args()))
