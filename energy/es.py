import argparse
import gym
import torch

from energy.models.mlp import ESMLP

from energy.tools.vec_env.subproc_vec_env import SubprocVecEnv

# from torch.distributions import Categorical

from utils.config import Config
# from utils.meter import Meter
from utils.log import Log


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        # is_atari = \
        #     isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = make_atari(env_id)
        env.seed(seed + rank)
        # if is_atari:
        #     env = wrap_deepmind(env, frame_stack=True)
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
        self._pool_size = config.get('energy_es_pool_size')

        self._envs = SubprocVecEnv(
            [make_env(
                self._config.get('energy_gym_env'),
                self._config.get('seed'),
                i,
            ) for i in range(self._pool_size)],
        )

        self._obs_shape = self._envs.observation_space.shape
        observation_size = self._envs.observation_space.shape[0]

        self._modules = [
            ESMLP(self._config, observation_size).to(self._device)
            for i in range(self._pool_size)
        ]

        Log.out('ES initialized', {
            "pool_size": self._config.get('energy_reinforce_pool_size'),
        })

        for m in self._modules:
            m.eval()

    def run_once(
            self,
            epoch: int,
    ):
        with torch.no_grad():
            noises = []

            for m in self._modules:
                noises += [m.noise()]
                m.apply_noise(noises[-1])

            import pdb; pdb.set_trace()

            all_done = False
            observations = self._env.reset()

            while(not all_done):
                for i, m in enumerate(self._modules):

            Log.out('FOO')


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
