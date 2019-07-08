import argparse
import gym
import os
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim

from energy.models.mlp import MLP
from energy.tools.atari_wrappers import make_atari, wrap_deepmind
from energy.tools.subproc_vec_env import SubprocVecEnv

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        is_atari = isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if is_atari:
            env = wrap_deepmind(env, frame_stack=True)
        return env

    return _thunk


class Rollouts:
    def __init__(
            self,
            config,
            obs_shape,
    ):
        self._config = config
        self._obs_shape = obs_shape

        self._device = torch.device(config.get('device'))

        self._gamma = config.get('energy_reinforce_gamma')
        self._tau = config.get('energy_reinforce_tau')

        self._rollout_size = config.get('energy_reinforce_rollout_size')
        self._pool_size = config.get('energy_reinforce_env_pool_size')
        self._batch_size = config.get('prooftrace_ppo_batch_size')

        self._observations = torch.zeros(
            self._rollout_size+1, self._pool_size, *(self._obs_shape),
        ).to(self._device)

        self._actions = torch.zeros(
            self._rollout_size, self._pool_size, 3,
            dtype=torch.int64,
        ).to(self._device)

        self._log_probs = torch.zeros(
            self._rollout_size, self._pool_size, 3,
        ).to(self._device)

        self._rewards = torch.zeros(
            self._rollout_size, self._pool_size, 1
        ).to(self._device)
        self._values = torch.zeros(
            self._rollout_size+1, self._pool_size, 1
        ).to(self._device)

        self._masks = torch.ones(
            self._rollout_size+1, self._pool_size, 1,
        ).to(self._device)

        self._returns = torch.zeros(
            self._rollout_size+1, self._pool_size, 1,
        ).to(self._device)

    def insert(
            self,
            step: int,
            observations,
            actions,
            log_probs,
            values,
            rewards,
            masks,
    ):
        self._observations[step+1].copy_(observations)
        self._actions[step].copy_(actions)
        self._log_probs[step].copy_(log_probs)
        self._values[step].copy_(values)
        self._rewards[step].copy_(rewards)
        self._masks[step+1].copy_(masks)

    def compute_returns(
            self,
            next_values,
    ):
        self._values[-1].copy_(next_values)
        self._returns[-1].copy_(next_values)

        # for step in reversed(range(self._rollout_size)):
        #     self._returns[step] = self._rewards[step] + \
        #         (self._gamma * self._returns[step+1] * self._masks[step+1])

        gae = 0
        for step in reversed(range(self._rollout_size)):
            delta = (
                self._rewards[step] +
                self._gamma * self._values[step+1] * self._masks[step+1] -
                self._values[step]
            )
            gae = delta + self._gamma * self._tau * self._masks[step+1] * gae
            self._returns[step] = gae + self._values[step]

    def after_update(
            self,
    ):
        self._observations[0].copy_(self._observations[-1])
        self._masks[0].copy_(self._masks[-1])

    def generator(
            self,
            advantages,
    ):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self._pool_size * self._rollout_size)),
            self._batch_size,
            drop_last=False,
        )
        for sample in sampler:
            indices = torch.LongTensor(sample).to(self._device)

            yield \
                self._observations[:-1].view(
                    -1, *(self._obs_shape),
                )[indices], \
                self._actions.view(-1, self._actions.size(-1))[indices], \
                self._values[:-1].view(-1, 1)[indices], \
                self._returns[:-1].view(-1, 1)[indices], \
                self._masks[:-1].view(-1, 1)[indices], \
                self._log_probs.view(-1, self._log_probs.size(-1))[indices], \
                advantages.view(-1, 1)[indices]


class A2C:
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
            config.get('energy_reinforce_learning_rate')

        Log.out(
            "Initializing A2C", {
                'paramater_count_MLP': self._model.parameters_count(),
            },
        )

        self._train_batch = 0

    def init_training(
            self,
    ):
        self._optimizer = optim.Adam(
            [
                {'params': self._model.parameters()},
            ],
            lr=self._learning_rate,
        )

        self._envs = SubprocVecEnv(
            [make_env(
                'PongNoFrameskip-v4', 
                self._config.get('seed'),
                i,
            ) for i in range(config.get(')],
        )
        obs_shape = self._envs.observation_space.shape

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
    os.environ['OMP_NUM_THREADS'] = '1'

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
        torch.cuda.manual_seed(config.get('seed'))

    torch.manual_seed(config.get('seed'))

    a2c = A2C(config)
    a2c.init_training()
    epoch = 0
    while True:
        a2c.batch_train(epoch)
        # lm.save()
        epoch += 1
