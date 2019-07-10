import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from math import ceil
import logging
import datetime
import os
from rl_baselines.environment import SubprocVecEnv


def set_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.handlers = []

    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    import os
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join("runs", current_time)

    os.makedirs(logdir, exist_ok=True)
    filename = os.path.join(logdir, "run.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logdir


logger = logging.getLogger("rl-baselines")
if not logger.handlers:
    logdir = set_logger(logger)


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


class Baseline:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _get(self, episodes):
        raise NotImplementedError

    def __call__(self, episodes):
        batch_weights = self._get(episodes)
        weights = torch.Tensor(batch_weights)
        if self.normalize:
            weights = (weights - weights.mean()) / (weights.std() + 1e-5)
        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"


class FullReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = [episode.ret for episode in episodes for _ in episode.rew]
        return weights


class FutureReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = []
        for episode in episodes:
            ret = 0
            returns = []
            for rew in reversed(episode.rew):
                ret += rew
                returns.append(ret)
            weights += list(reversed(returns))
        return weights


def discounted_cumsum(elements, gamma):
    discounted = []
    cur = 0
    for element in reversed(elements):
        cur = element + gamma * cur
        discounted.append(cur)
    return list(reversed(discounted))


class DiscountedReturnBaseline(Baseline):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def _get(self, episodes):
        weights = []
        for episode in episodes:
            weights += discounted_cumsum(episode.rew, self.gamma)
        return weights


class GAEBaseline(Baseline):
    def __init__(self, value_model, gamma, lambda_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value_model = value_model
        assert gamma > lambda_, f"For stability λ({lambda_}) < γ({gamma}) is advised"

    def _get(self, episodes):
        obs = [item for episode in episodes for item in episode.obs]
        with torch.no_grad():
            values = self.value_model(torch.Tensor(obs))
        start = 0
        weights = []
        for episode in episodes:
            end = start + len(episode.obs)
            # End value always 0 as we never cut episodes.
            v_pi = torch.cat((values[start:end], torch.Tensor([[0]])), dim=0)
            deltas = (
                torch.Tensor(episode.rew).unsqueeze(1)
                + self.gamma * v_pi[1:]
                - v_pi[:-1]
            )
            assert list(deltas.shape) == [len(episode.obs), 1], "Dimension sanity check"
            weights += discounted_cumsum(deltas, self.gamma * self.lambda_)
            start = end
        return torch.Tensor(weights)


class ModelUpdate(nn.Module):
    loss_name = "loss"

    def __init__(self, model, optimizer, baseline, iters=1):
        super().__init__()
        self.baseline = baseline
        self.model = model
        self.optimizer = optimizer
        self.iters = iters

    def update(self, episodes):
        raise NotImplementedError

    def batch(self, episodes):
        batch_obs = [item for episode in episodes for item in episode.obs]
        batch_acts = [item for episode in episodes for item in episode.act]
        weights = self.baseline(episodes)
        obs = torch.Tensor(batch_obs)
        acts = torch.stack(batch_acts, dim=0)
        return obs, acts, weights


class PolicyUpdate(ModelUpdate):
    loss_name = "policy_loss"

    @property
    def policy(self):
        return self.model


class ValueUpdate(ModelUpdate):
    loss_name = "value_loss"

    @property
    def value(self):
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.baseline, DiscountedReturnBaseline
        ), "Value models need to learn discounted returns"

    def update(self, episodes):
        obs, _, returns = self.batch(episodes)

        for i in range(self.iters):
            values = self.model(obs)
            loss = F.mse_loss(values, returns.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {self.loss_name: loss}


class ActorCriticUpdate(nn.Module):
    def __init__(self, policy_update, value_update):
        super().__init__()
        assert isinstance(policy_update, PolicyUpdate)
        assert isinstance(value_update, ValueUpdate)
        self.policy_update = policy_update
        self.value_update = value_update

    @property
    def policy(self):
        return self.policy_update.policy

    @property
    def value(self):
        return self.value_update.value

    def update(self, episodes):
        losses = {}
        p_losses = self.policy_update.update(episodes)
        v_losses = self.value_update.update(episodes)
        losses.update(p_losses)
        losses.update(v_losses)
        return losses


class Episode:
    def __init__(self):
        self.obs = []
        self.act = []
        self.rew = []

    def end(self):
        self.ret = sum(self.rew)
        self.len = len(self.rew)
        assert len(self.rew) == len(self.obs) == len(self.act)

    def __len__(self):
        return self.len


def gather_episodes(env, batch_size, policy):
    episodes = []

    num_envs = env.num_envs

    batch_episodes = [Episode() for i in range(num_envs)]

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    step = 0
    stops = [False for i in range(num_envs)]

    while True:
        for i in range(num_envs):
            if stops[i]:
                continue
            batch_episodes[i].obs.append(obs[i])

        # act in the environment
        x = torch.from_numpy(obs).float()
        dist = policy(x)
        act = dist.sample()
        obs, rew, done, _ = env.step(act.cpu().numpy())

        # save action, reward
        for i in range(num_envs):
            if stops[i]:
                continue
            batch_episodes[i].act.append(act[i])
            batch_episodes[i].rew.append(rew[i])

            if done[i]:
                # if episode is over, record info about episode
                batch_episodes[i].end()
                episodes.append(batch_episodes[i])

                # reset episode-specific variables
                spec_obs = env.reset(i)
                obs[i] = spec_obs
                batch_episodes[i] = Episode()

                if step > batch_size:
                    stops[i] = True
                    if all(stops):
                        return episodes
        step += num_envs - sum(stops)
    return episodes


def train_one_epoch(env, batch_size, render, policy_update, device):

    policy = policy_update.policy
    # collect experience by acting in the environment with current policy
    # start = datetime.datetime.now()

    episodes = gather_episodes(env, batch_size, policy)

    # end_rollout = datetime.datetime.now()
    losses = policy_update.update(episodes)

    # logger.debug(f"Rollout time {end_rollout - start}")
    # logger.debug(f"Update time {datetime.datetime.now() - end_rollout}")

    return losses, episodes


def default_model(env, hidden_sizes, n_acts):
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, (Discrete, Box)
    ), "This example only works for envs with discrete/box action spaces."

    assert len(env.observation_space.shape) in [
        1,
        3,
    ], f"This example only works for envs with Box(n,) or Box(h, w, c) not {env.observation_space} observation spaces."
    obs_dim = env.observation_space.shape[0]

    if len(env.observation_space.shape) == 1:
        obs_dim = env.observation_space.shape[0]
        model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
    elif len(env.observation_space.shape) == 3:
        model = Conv(
            input_shape=env.observation_space.shape, sizes=hidden_sizes + [n_acts]
        )
    return model


def create_models(env_name, num_envs, hidden_sizes, pi_lr, vf_lr):
    # make environment, check spaces, get obs / act dims
    env = SubprocVecEnv([lambda: gym.make(env_name) for i in range(num_envs)])

    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
    elif isinstance(env.action_space, Box):
        assert (
            len(env.action_space.shape) == 1
        ), f"This example only works for envs with Box(n,) not {env.action_space} action spaces."
        n_acts = env.action_space.shape[0]
    model = default_model(env, hidden_sizes, n_acts)
    if isinstance(env.action_space, Discrete):
        policy = DiscretePolicy(model)
    elif isinstance(env.action_space, Box):
        policy = ContinuousPolicy(model, env.action_space.shape)

    else:
        raise NotImplementedError(
            "We don't handle action spaces different from box/discrete yet."
        )
    poptimizer = torch.optim.Adam(policy.parameters(), lr=pi_lr)

    value = default_model(env, hidden_sizes, 1)
    voptimizer = torch.optim.Adam(value.parameters(), lr=vf_lr)
    return env, (policy, poptimizer), (value, voptimizer)


def solve(
    env_name,
    env,
    policy_update,
    logdir,
    epochs=100,
    batch_size=5000,
    render=False,
    device=None,
):
    if device is None:
        device = "cpu"

    writer = SummaryWriter(log_dir=logdir)
    env_step = 0

    # Weird bug, tensorboard sets its own root logger, we need to remove it.
    root = logging.getLogger()
    root.handlers = []

    parameters = sum(p.numel() for p in policy_update.parameters())
    logger.debug(f"Attempting to solve {env_name}")
    logger.debug(f"Epochs: {epochs}")
    logger.debug(f"Batch_size: {batch_size}")
    logger.debug(f"Policy Update: {policy_update}")
    logger.debug(f"Parameters: {parameters}")
    logger.debug(f"Reward threshold: {env.spec.reward_threshold}")

    max_ret = -1e9

    for epoch in range(epochs):
        losses, episodes = train_one_epoch(
            env, batch_size, render, policy_update=policy_update, device=device
        )
        rets = np.mean([episode.ret for episode in episodes])
        lens = np.mean([episode.len for episode in episodes])
        loss_string = "\t".join(
            [f"{loss_name}: {loss:.3f}" for loss_name, loss in losses.items()]
        )
        logger.debug(
            "epoch: %3d \t %s \t return: %.3f \t ep_len: %.3f"
            % (epoch, loss_string, rets, lens)
        )
        env_step += sum([episode.len for episode in episodes])
        writer.add_scalar(f"{env_name}/episode_reward", rets, global_step=env_step)
        writer.add_scalar(f"{env_name}/episode_length", lens, global_step=env_step)
        if rets > max_ret:
            filename = os.path.join(logdir, "checkpoint.pth")
            torch.save(policy_update, filename)
            logger.debug(f"Saved new best model: {filename}")
            max_ret = rets
        if env.spec.reward_threshold and rets > env.spec.reward_threshold:
            logger.info(f"{env_name}: Solved !")
            logger.info(
                f"{env_name}: Check out winning agent `python -m rl_baselines.test_agent --model={filename} --env={env_name}`"
            )
            return True
    return False if env.spec.reward_threshold else None
