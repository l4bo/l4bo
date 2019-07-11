import torch
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


class ESMLP(nn.Module):
    def __init__(
            self,
            config,
            observation_size,
            action_size,
    ):
        super().__init__()

        self._config = config

        self._device = torch.device(config.get('device'))
        self._hidden_size = config.get('energy_hidden_size')

        self._l1 = nn.Linear(observation_size, self._hidden_size, bias=False)
        self._l2 = nn.Linear(self._hidden_size, action_size, bias=False)

        self._mlp = nn.Sequential(
            self._l1,
            nn.ReLU(),
            self._l2,
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
        out = self._mlp(observations)

        return out
