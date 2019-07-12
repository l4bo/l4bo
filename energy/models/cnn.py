import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
            self,
            config,
            channel_count,
    ):
        super().__init__()

        self._hidden_size = config.get('energy_hidden_size')

        self._convs = nn.Sequential(
            nn.Conv2d(channel_count, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, 4, stride=2, padding=1),
            # nn.ReLU(),
        )

        self._tail = nn.Sequential(
            nn.Linear(32*7*7, self._hidden_size),
            nn.ReLU(),
            # nn.LayerNorm(self._hidden_size),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            observations,
    ):
        out = self._convs(observations)

        out = out.view(-1, 32*7*7)
        assert out.size(0) == observations.size(0)

        hiddens = self._tail(out)

        return hiddens


class ESCNN(nn.Module):
    def __init__(
            self,
            config,
            channel_count,
            action_size,
    ):
        super().__init__()

        self._config = config

        self._device = torch.device(config.get('device'))
        self._hidden_size = config.get('energy_hidden_size')

        self._convs = nn.Sequential(
            nn.Conv2d(channel_count, 32, 8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        self._tail = nn.Sequential(
            nn.Linear(64, self._hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(self._hidden_size, action_size, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def zero_noise(
            self,
    ):
        noise = {}
        for name, param in self.named_parameters():
            noise[name] = self._config.get('energy_es_sigma') * \
                torch.zeros(param.size()).to(self._device)
        return noise

    def noise(
            self,
    ):
        noise = {}
        for name, param in self.named_parameters():
            noise[name] = self._config.get('energy_es_sigma') * \
                torch.randn(param.size()).to(self._device)
        return noise

    def apply_noise(
            self,
            noise,
    ):
        for name, param in self.named_parameters():
            param.data += noise[name]

    def revert_noise(
            self,
            noise,
    ):
        for name, param in self.named_parameters():
            param.data -= noise[name]

    def forward(
            self,
            observations,
    ):
        hiddens = self._convs(observations)
        outs = self._tail(hiddens.squeeze().unsqueeze(0))

        return outs
