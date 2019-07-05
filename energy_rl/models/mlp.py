import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class MLP(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(MLP, self).__init__()

        self._device = torch.device(config.get('device'))

        self.hidden_size = config.get('energy_rl_mlp_hidden_size')
        self.layer_count = config.get('energy_rl_mlp_layer_count')

        self._layers = nn.ModuleList()
        for _ in range(self.layer_count):
            layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.layers.append(layer)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            x: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = F.relu(layer(x))

        # TODO compute consumption.
        c = torch.zeros(x.size(0), 1)

        return x, c
