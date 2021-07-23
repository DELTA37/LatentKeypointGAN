import argparse
import torch
from torchvision import transforms
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
def test_spade_generator(batch_size=16,
                         noise_dim=64):
    generator = SPADEGenerator(noise_dim=noise_dim)
    generator.eval()
    z = torch.randn(batch_size, 3 * noise_dim)
    out, latents = generator(z, return_latents=True)
    print(latents.shape)
    print(out.shape)

    for b in range(batch_size):
        transforms.ToPILImage()(out[0]).show()


if __name__ == '__main__':
    test_spade_generator(**vars(parse_args()))
