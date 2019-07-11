import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter


class VAEDataset(Dataset):
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise Exception("Try to create some frames first")
        self.data = torch.load(filename)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        result = RESIZE(self.data[i])
        return result


class Decoder(nn.Module):
    """ VAE decoder """

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        N = 1024
        self.fc1 = nn.Linear(latent_size, N)
        self.deconv1 = nn.ConvTranspose2d(N, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """

    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        # self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        N = 1024
        self.fc_mu = nn.Linear(N, latent_size)
        self.fc_logsigma = nn.Linear(N, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """

    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


RESIZE = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()]
)


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """

    BCE = F.mse_loss(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    loss = BCE + 0 * KLD
    return loss, {"loss": loss, "BCE": BCE, "KLD": KLD}


def main():

    batch_size = 32
    epochs = 100
    lr = 1e-4
    latent_size = 1024
    time_channels = 4
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filename = "frames.pty"
    dataset = VAEDataset(filename)
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    vae_model = VAE(time_channels, latent_size)
    vae_model.to(device)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        torch.optim.SGD(vae_model.parameters(), lr=lr, momentum=0.9),
        base_lr=0,
        max_lr=lr,
    )

    writer = SummaryWriter()

    step = 0
    best_loss = 1e9
    for epoch in tqdm.tqdm(range(epochs)):
        avg_loss = 0
        for item in tqdm.tqdm(dataloader):
            item = item.to(device)
            reconstructed, mu, logsigma = vae_model(item)
            loss, losses = loss_function(reconstructed, item, mu, logsigma)

            for loss_name, l in losses.items():
                writer.add_scalar(f"VAE/{loss_name}", l, global_step=step)

            scheduler.optimizer.zero_grad()
            loss.backward()
            scheduler.optimizer.step()
            scheduler.step()
            step += 1
            avg_loss += loss.item()
        avg_loss /= len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving best model {avg_loss}")
            torch.save(vae_model, "vae.pth")


if __name__ == "__main__":
    main()
