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

        self._device = torch.device(config.get('device'))
        self._save_dir = config.get('save_dir')
        self._load_dir = config.get('load_dir')

        self._model = MLP(config).to(self._device)
        self._learning_rate = \
            config.get('energy_rl_reinforce_learning_rate')

        Log.out(
            "Initializing REINFORCE", {
                'paramater_count_MLP': self._model.parameters_count(),
            },
        )

        self._train_batch = 0

    def init_training(
            self,
            env: str,
    ):
        self._optimizer = optim.Adam(
            [
                {'params': self._model.parameters()},
            ],
            lr=self._learning_rate,
        )

        # TODO setup gym env in self._env

    def load(
            self,
            training=True,
    ):
        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model.pt",
            ):
                Log.out(
                    "Loading models", {
                        'load_dir': self._load_dir,
                    })
                self._mode.load_state_dict(
                    torch.load(
                        self._load_dir + "/model.pt",
                        map_location=self._device,
                    ),
                )
                if training:
                    if os.path.isfile(
                            self._load_dir + "/optimizer.pt",
                    ):
                        self._optimizer.load_state_dict(
                            torch.load(
                                self._load_dir + "/optimizer.pt",
                                map_location=self._device,
                            ),
                        )

        return self

    def save(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._model.state_dict(),
                self._save_dir + "/model.pt",
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer.pt",
            )

    def batch_train(
            self,
            epoch: int,
    ):
        assert self._optimizer is not None

        self._model.train()

        # TRAIN LOOP

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.load_dir is not None:
        config.override(
            'load_dir',
            os.path.expanduser(args.load_dir),
        )
    if args.save_dir is not None:
        config.override(
            'save_dir',
            os.path.expanduser(args.save_dir),
        )
    if args.device is not None:
        config.override('device', args.device)

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    torch.manual_seed(0)

    reinforce = REINFORCE(config)
    reinforce.init_training()
    epoch = 0
    while True:
        reinforce.batch_train(epoch)
        # lm.save()
        epoch += 1
