import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            config,
            observation_size,
    ):
        super().__init__()

        self._hidden_size = config.get('energy_hidden_size')

        self._mlp = nn.Sequential(
            nn.Linear(observation_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            observations,
    ):
        hiddens = self._mlp(observations)

        return hiddens
