import torch.nn as nn


class PH(nn.Module):
    def __init__(
            self,
            config,
            action_size,
    ):
        super().__init__()

        self._hidden_size = config.get('energy_hidden_size')

        self._action_head = nn.Sequential(
            nn.Linear(self._hidden_size, action_size),
            nn.LogSoftmax(dim=1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            hiddens,
    ):
        actions = self._action_head(hiddens)

        return actions


class VH(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()

        self._hidden_size = config.get('energy_hidden_size')

        self._value_head = nn.Sequential(
            nn.Linear(self._hidden_size, 1),
            nn.Softplus(),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            hiddens,
    ):
        values = self._value_head(hiddens)

        return values
