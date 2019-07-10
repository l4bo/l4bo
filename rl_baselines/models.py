import torch.nn as nn
from math import ceil
import torch
from torch.distributions import Categorical, MultivariateNormal


class Conv(nn.Module):
    def __init__(
        self, input_shape, sizes, activation=nn.ReLU(inplace=True), out_activation=None
    ):
        super().__init__()

        self.input_shape = input_shape
        self.activation = activation
        H, W, C = input_shape

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip([C] + sizes, sizes):
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

        h = H
        w = W
        for i in range(len(sizes)):
            h /= 2
            w /= 2
            h = int(ceil(h))
            w = int(ceil(w))

        in_ = h * w * sizes[-1]
        self.out = nn.Linear(in_, sizes[-1])
        self.out_activation = out_activation

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        assert len(x.shape) == 4
        B, H, W, C = x.shape
        # Pytorch uses C, H, W for its convolution
        x = x.permute(0, 3, 1, 2)
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)

        # Flatten
        x = x.reshape(B, -1)
        x = self.out(x)
        if self.out_activation:
            x = self.out_activate(x)
        return x


class MLP(nn.Module):
    def __init__(self, sizes, activation=torch.tanh, out_activation=None):
        super().__init__()

        self.layers = nn.ModuleList()
        for in_, out_ in zip(sizes, sizes[1:]):
            layer = nn.Linear(in_, out_)
            self.layers.append(layer)
        self.activation = activation
        self.out_activation = out_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        if self.out_activation:
            x = self.out_activation(x)
        return x


class DiscretePolicy(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.model = policy_model

    def forward(self, state):
        logits = self.model(state)
        return Categorical(logits=logits)


class ContinuousPolicy(nn.Module):
    def __init__(self, policy_model, action_shape):
        super().__init__()
        self.model = policy_model

        # log_std = -0.5 -> std=0.6
        self.log_std = nn.Parameter(-0.5 * torch.ones(*action_shape))

    def forward(self, state):
        mu = self.model(state)
        return MultivariateNormal(mu, torch.diag(self.log_std.exp()))
