import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import datetime
import os
import sys
from rl_baselines.environment import SubprocVecEnv, make_single_env
from rl_baselines.models import ContinuousPolicy, DiscretePolicy, MLP, Conv


def set_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.handlers = []

    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
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

        # env.render()

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


def make_env(env_name, num_envs):
    env = SubprocVecEnv([lambda: make_single_env(env_name) for i in range(num_envs)])
    return env


def create_models(env, hidden_sizes, pi_lr, vf_lr):
    # make environment, check spaces, get obs / act dims

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
    return (policy, poptimizer), (value, voptimizer)


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
