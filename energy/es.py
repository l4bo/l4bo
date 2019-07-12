import argparse
import numpy as np
import gym
import torch

from energy.models.mlp import ESMLP
from energy.models.cnn import ESCNN

from energy.tools.atari_wrappers import make_atari, wrap_deepmind
from energy.tools.vec_env.subproc_vec_env import SubprocVecEnv

from torch.distributions import Categorical

from utils.config import Config
# from utils.meter import Meter
from utils.log import Log


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        if env_id == "PongNoFrameskip-v":
            env = make_atari(env_id)
            env = wrap_deepmind(env, frame_stack=True)
        return env
    return _thunk


class ES:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._alpha = config.get('energy_es_alpha')
        self._sigma = config.get('energy_es_sigma')
        self._pool_size = config.get('energy_es_pool_size')

        self._envs = SubprocVecEnv(
            [make_env(
                self._config.get('energy_gym_env'),
                self._config.get('seed'),
                i,
            ) for i in range(self._pool_size)],
        )

        self._image_input = False
        if len(self._envs.observation_space.shape) == 3:
            self._image_input = True
        else:
            assert len(self._envs.observation_space.shape) == 1

        if self._image_input:
            self._obs_shape = (
                self._envs.observation_space.shape[2],
                self._envs.observation_space.shape[1],
                self._envs.observation_space.shape[0],
            )
            channel_count = self._envs.observation_space.shape[2]
        else:
            self._obs_shape = self._envs.observation_space.shape
            self._observation_size = self._envs.observation_space.shape[0]
        self._action_size = self._envs.action_space.n

        if self._image_input:
            self._modules = [
                ESCNN(
                    self._config, channel_count, self._action_size,
                ).to(self._device)
                for i in range(self._pool_size)
            ]
        else:
            self._modules = [
                ESMLP(
                    self._config, self._observation_size, self._action_size,
                ).to(self._device)
                for i in range(self._pool_size)
            ]

        Log.out('ES initialized', {
            "pool_size": self._config.get('energy_es_pool_size'),
        })

        for m in self._modules:
            m.train()

        self._smoothed_reward = None

    def run_once(
            self,
            epoch: int,
    ):
        with torch.no_grad():
            observations = self._envs.reset()

            if self._image_input:
                obs = torch.from_numpy(
                    observations,
                ).float().transpose(3, 1).to(self._device) / 255.0
            else:
                obs = torch.from_numpy(
                    observations,
                ).float().to(self._device)

            noises = []
            for m in self._modules:
                n = m.noise()
                m.apply_noise(n)
                noises += [n]

            pool_dones = [False] * self._pool_size
            pool_rewards = [0.0] * self._pool_size

            all_done = False
            while(not all_done):
                actions = []
                for i, m in enumerate(self._modules):
                    prd_actions = m(obs[i].unsqueeze(0))
                    m = Categorical(torch.exp(prd_actions))
                    actions += [m.sample().cpu().numpy()]

                observations, rewards, dones, infos = self._envs.step(
                    np.squeeze(np.array(actions), 1),
                )
                if self._image_input:
                    obs = torch.from_numpy(
                        observations,
                    ).float().transpose(3, 1).to(self._device) / 255.0
                else:
                    obs = torch.from_numpy(
                        observations,
                    ).float().to(self._device)

                all_done = True
                for i in range(self._pool_size):
                    if not pool_dones[i] or (not pool_dones[i] and dones[i]):
                        pool_rewards[i] += rewards[i]
                    pool_dones[i] = pool_dones[i] or dones[i]
                    all_done = all_done and pool_dones[i]

            rewards = torch.tensor(pool_rewards)
            advantages = (rewards - torch.mean(rewards)) \
                / (torch.std(rewards) + 10e-7)

            final = self._modules[0].zero_noise()
            for i in range(self._pool_size):
                for name in final.keys():
                    final[name] += noises[i][name] * advantages[i] * \
                        self._alpha / (self._pool_size * self._sigma)

            for i, m in enumerate(self._modules):
                m.revert_noise(noises[i])
                m.apply_noise(final)

            if self._smoothed_reward is None:
                self._smoothed_reward = rewards.mean()
            self._smoothed_reward = \
                0.95 * self._smoothed_reward + 0.05 * rewards.mean()

            Log.out("ENERGY ES RUN", {
                'epoch': epoch,
                'reward': "{:.4f}".format(rewards.mean()),
                'smoothed': "{:.4f}".format(self._smoothed_reward),
            })


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

    es = ES(config)

    epoch = 0
    while True:
        es.run_once(epoch)
        epoch += 1
