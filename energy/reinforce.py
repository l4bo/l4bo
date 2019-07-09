import argparse
import os
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from energy.models.cnn import CNN
from energy.models.heads import PH, VH

from energy.tools.atari_wrappers import make_atari, wrap_deepmind
from energy.tools.vec_env.subproc_vec_env import SubprocVecEnv

from torch.distributions import Categorical
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
        self._pool_size = config.get('energy_reinforce_pool_size')

        self._observations = torch.zeros(
            self._rollout_size+1, self._pool_size, *(self._obs_shape),
        ).to(self._device)

        self._actions = torch.zeros(
            self._rollout_size, self._pool_size, 1,
            dtype=torch.int64,
        ).to(self._device)

        self._log_probs = torch.zeros(
            self._rollout_size, self._pool_size, 1,
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
            batch_size,
    ):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self._pool_size * self._rollout_size)),
            batch_size,
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

    def batch(
            self,
            advantages,
    ):
        return (
            self._observations[:-1].view(
                -1, *(self._obs_shape),
            ),
            self._actions.view(-1, self._actions.size(-1)),
            self._values[:-1].view(-1, 1),
            self._returns[:-1].view(-1, 1),
            self._masks[:-1].view(-1, 1),
            self._log_probs.view(-1, self._log_probs.size(-1)),
            advantages.view(-1, 1),
        )


class A2C:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))
        self._save_dir = config.get('save_dir')
        self._load_dir = config.get('load_dir')

        self._learning_rate = config.get('energy_reinforce_learning_rate')
        self._rollout_size = config.get('energy_reinforce_rollout_size')
        self._pool_size = config.get('energy_reinforce_pool_size')
        self._value_coeff = config.get('energy_reinforce_value_coeff')
        self._entropy_coeff = config.get('energy_reinforce_entropy_coeff')
        self._grad_norm_max = config.get('energy_reinforce_grad_norm_max')

        self._envs = SubprocVecEnv(
            [make_env(
                self._config.get('energy_gym_env'),
                self._config.get('seed'),
                i,
            ) for i in range(self._pool_size)],
        )

        self._obs_shape = (
            self._envs.observation_space.shape[2],
            self._envs.observation_space.shape[0],
            self._envs.observation_space.shape[1],
        )
        channel_count = self._envs.observation_space.shape[2]

        self._rollouts = Rollouts(self._config, self._obs_shape)

        observations = self._envs.reset()

        self._rollouts._observations[0].copy_(
            torch.from_numpy(
                observations,
            ).float().transpose(3, 1).to(self._device) / 255.0,
        )
        self._episode_rewards = [0.0] * self._pool_size

        self._modules = {
            'CNN': CNN(self._config, channel_count).to(self._device),
            'PH': PH(self._config, self._envs.action_space.n).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        Log.out(
            "Initializing A2C modules", {
                'paramater_count_CNN': self._modules['CNN'].parameters_count(),
                'paramater_count_PH': self._modules['PH'].parameters_count(),
                'paramater_count_VH': self._modules['VH'].parameters_count(),
            },
        )

        self._optimizer = optim.Adam(
            [
                {'params': self._modules['CNN'].parameters()},
                {'params': self._modules['PH'].parameters()},
                {'params': self._modules['VH'].parameters()},
            ],
            lr=self._learning_rate,
        )

        Log.out('A2C initialized', {
            "pool_size": self._config.get('energy_reinforce_pool_size'),
            "rollout_size": self._config.get('energy_reinforce_rollout_size'),
        })

    def load(
            self,
            training=True,
    ):
        super().load()

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model.pt",
            ):
                Log.out(
                    "Loading models", {
                        'load_dir': self._load_dir,
                    })
                self._modules['CNN'].load_state_dict(
                    torch.load(
                        self._load_dir + "/module_CNN.pt",
                        map_location=self._device,
                    ),
                )
                self._modules['PH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/module_PH.pt",
                        map_location=self._device,
                    ),
                )
                self._modules['VH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/module_VH.pt",
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
                self._modules['CNN'].state_dict(),
                self._save_dir + "/module_CNN.pt",
            )
            torch.save(
                self._modules['PH'].state_dict(),
                self._save_dir + "/module_PH.pt",
            )
            torch.save(
                self._modules['VH'].state_dict(),
                self._save_dir + "/module_VH.pt",
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer.pt",
            )

    def run_once(
            self,
            epoch: int,
    ):
        assert self._optimizer is not None

        for m in self._modules:
            self._modules[m].train()

        reward_meter = Meter()
        act_loss_meter = Meter()
        val_loss_meter = Meter()
        entropy_meter = Meter()

        for step in range(self._rollout_size):
            with torch.no_grad():
                obs = self._rollouts._observations[step]

                hiddens = self._modules['CNN'](obs).detach()
                prd_actions = self._modules['PH'](hiddens)
                values = self._modules['VH'](hiddens)

                m = Categorical(torch.exp(prd_actions))
                actions = m.sample().view(-1, 1)

                observations, rewards, dones, infos = self._envs.step(
                    actions.cpu().numpy(),
                )

                observations = torch.from_numpy(
                    observations,
                ).float().transpose(3, 1).to(self._device) / 255.0

                log_probs = prd_actions.gather(1, actions)

                for i, r in enumerate(rewards):
                    self._episode_rewards[i] += r
                    if dones[i]:
                        reward_meter.update(self._episode_rewards[i])
                        self._episode_rewards[i] = 0.0

                self._rollouts.insert(
                    step,
                    observations,
                    actions.detach(),
                    log_probs.detach(),
                    values.detach(),
                    torch.tensor(
                        [r for r in rewards], dtype=torch.float,
                    ).unsqueeze(1).to(self._device),
                    torch.tensor(
                        [[0.0] if d else [1.0] for d in dones],
                    ).to(self._device),
                )

        with torch.no_grad():
            obs = self._rollouts._observations[-1]

            hiddens = self._modules['CNN'](obs).detach()
            values = self._modules['VH'](hiddens)

            self._rollouts.compute_returns(values.detach())

            advantages = \
                self._rollouts._returns[:-1] - self._rollouts._values[:-1]
            advantages = \
                (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        rollout_observations, \
            rollout_actions, \
            rollout_values, \
            rollout_returns, \
            rollout_masks, \
            rollout_log_probs, \
            rollout_advantages = self._rollouts.batch(advantages)

        hiddens = self._modules['CNN'](rollout_observations)
        prd_actions = self._modules['PH'](hiddens)
        values = self._modules['VH'](hiddens)

        log_probs = prd_actions.gather(1, rollout_actions)
        entropy = -(
            (prd_actions * torch.exp(prd_actions)).mean()
        )

        action_loss = -(rollout_advantages * log_probs).mean()
        value_loss = F.mse_loss(values, rollout_returns)

        # Backward pass.
        self._optimizer.zero_grad()

        (
            action_loss +
            self._value_coeff * value_loss -
            self._entropy_coeff * entropy
        ).backward()

        if self._grad_norm_max > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._modules['CNN'].parameters(),
                self._grad_norm_max,
            )
            torch.nn.utils.clip_grad_norm_(
                self._modules['PH'].parameters(),
                self._grad_norm_max,
            )
            torch.nn.utils.clip_grad_norm_(
                self._modules['VH'].parameters(),
                self._grad_norm_max,
            )

        self._optimizer.step()

        act_loss_meter.update(action_loss.item())
        val_loss_meter.update(value_loss.item())
        entropy_meter.update(entropy.item())

        self._rollouts.after_update()

        Log.out("ENERGY A2C RUN", {
            'epoch': epoch,
            'reward': "{:.4f}".format(reward_meter.avg or 0.0),
            'act_loss': "{:.4f}".format(act_loss_meter.avg or 0.0),
            'val_loss': "{:.4f}".format(val_loss_meter.avg or 0.0),
            'entropy': "{:.4f}".format(entropy_meter.avg or 0.0),
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

    epoch = 0
    while True:
        a2c.run_once(epoch)
        # lm.save()
        epoch += 1
