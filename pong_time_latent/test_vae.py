import torch
from pong_time_latent.train_vae import *
import matplotlib.pyplot as plt
from random import randrange


def show(dataset, model, M):
    first_images = dataset[M]
    x = first_images.unsqueeze(0)
    reconstructed, _, _ = model(x.cuda())
    for N in range(4):
        plt.subplot(2, 4, N + 1)
        plt.imshow(first_images[N].detach().numpy(), cmap="gray")

        plt.subplot(2, 4, 4 + N + 1)
        plt.imshow(reconstructed[0, N].cpu().detach(), cmap="gray")

    plt.show()


def main():
    model = torch.load("vae.pth")
    dataset = VAEDataset("frames.pty")

    n = randrange(len(dataset))
    show(dataset, model, n)


if __name__ == "__main__":
    main()
