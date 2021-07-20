import argparse
import torch
from latent_kp_gan.nets import SPADEGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--noise_dim",
                        default=64,
                        type=int)
    args = parser.parse_args()
    return args


@torch.no_grad()
def test_spade_generator(batch_size=4,
                         noise_dim=512):
    generator = SPADEGenerator(noise_dim=noise_dim)
    z = torch.randn(batch_size, 3 * noise_dim)
    out = generator(z)
    print(out.shape)


if __name__ == '__main__':
    test_spade_generator(**vars(parse_args()))
