import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim

from energy_rl.models.mlp import MLP

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


class REINFORCE:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._model = MLP(config).to(self._device)
        self._learning_rate = \
            config.get('energy_rl_reinforce__learning_rate')

        Log.out(
            "Initializing REINFORCE", {
                'paramater_count_MLP': self._model.parameters_count(),
            },
        )

        self._train_batch = 0

    def init_training(
            self,
            env: string,
    ):
        self._optimizer = optim.Adam(
            [
                {'params': self._model_E.parameters()},
                {'params': self._model_H.parameters()},
                {'params': self._model_PH.parameters()},
                {'params': self._model_VH.parameters()},
            ],
            lr=self._learning_rate,
        )
