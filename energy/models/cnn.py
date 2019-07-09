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
