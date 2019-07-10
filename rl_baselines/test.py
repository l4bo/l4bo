import unittest
import logging
import multiprocessing
import torch
from rl_baselines.core import create_models, logdir, solve, logger, Episodes, make_env
from rl_baselines.baselines import (
    FutureReturnBaseline,
    DiscountedReturnBaseline,
    GAEBaseline,
)
from rl_baselines.reinforce import REINFORCE
from rl_baselines.ppo import PPO


logger.setLevel(logging.CRITICAL)


def discounted_cumsum(elements, gamma):
    discounted = []
    cur = 0
    for element in reversed(elements):
        cur = element + gamma * cur
        discounted.append(cur)
    return list(reversed(discounted))


class TestBaselines(unittest.TestCase):
    def setUp(self):
        episodes = Episodes(num_env=2, num_steps=4, obs_shape=(2,))
        # There is one last observation for advantage estimation.
        episodes.obs[0] = torch.Tensor([[1, -2], [1, -2], [1, -2], [1, -2], [1, -2]])
        episodes.rews[0] = torch.Tensor([1, 0, 1, 0])
        episodes.dones[0] = torch.Tensor([0, 0, 0, 0])
        episodes.acts[0] = torch.Tensor([1, 1, 1, 1])

        episodes.obs[1] = torch.Tensor(
            [[10, -20], [10, -20], [10, -20], [-10, 20], [-10, 20]]
        )
        episodes.rews[1] = torch.Tensor([1, 1, 1, 1])
        episodes.acts[1] = torch.Tensor([1, 1, 1, 1])
        episodes.dones[1] = torch.Tensor([0, 1, 0, 0])

        self.episodes = episodes

    def test_future_return_baseline(self):
        baseline = FutureReturnBaseline(normalize=False)
        weights = baseline(self.episodes)

        target = torch.Tensor([[2.0, 1.0, 1.0, 0], [1.0, 3.0, 2.0, 1.0]])
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = FutureReturnBaseline(normalize=True)
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())

    def test_discounted_return_baseline(self):
        gamma = 0.99
        baseline = DiscountedReturnBaseline(gamma=gamma, normalize=False)
        weights = baseline(self.episodes)

        target = torch.Tensor(
            [
                [1.0 + 1.0 * gamma ** 2, 0.0 + 1.0 * gamma, 1.0, 0.0],
                [1.0, 1.0 + 1.0 * gamma + 1 * gamma ** 2, 1.0 + 1.0 * gamma, 1.0],
            ]
        )
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = DiscountedReturnBaseline(gamma=gamma, normalize=True)
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())

    def test_gae_return_baseline(self):
        gamma = 0.99
        lambda_ = 0.95
        values = torch.arange(1, 11).reshape((2, 5)).float()
        value_model = lambda x: values
        baseline = GAEBaseline(
            value_model=value_model, gamma=gamma, lambda_=lambda_, normalize=False
        )
        weights = baseline(self.episodes)

        rewards = self.episodes.rews
        deltas = torch.Tensor(
            [
                [
                    rewards[0, 0] + (values[0, 1] * gamma - values[0, 0]),
                    rewards[0, 1] + (values[0, 2] * gamma - values[0, 1]),
                    rewards[0, 2] + (values[0, 3] * gamma - values[0, 2]),
                    rewards[0, 3] + (values[0, 4] * gamma - values[0, 3]),
                ],
                [
                    # XXX: cut episode here
                    rewards[1, 0] + (-values[1, 0]),
                    rewards[1, 1] + (values[1, 2] * gamma - values[1, 1]),
                    rewards[1, 2] + (values[1, 3] * gamma - values[1, 2]),
                    rewards[1, 3] + (values[1, 4] * gamma - values[1, 3]),
                ],
            ]
        )
        target = torch.Tensor(
            [
                discounted_cumsum(deltas[0], gamma * lambda_),  # First episode
                discounted_cumsum(deltas[1, :1], gamma * lambda_)
                + discounted_cumsum(deltas[1, 1:], gamma * lambda_),  # Second episode
            ]
        )
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = GAEBaseline(
            value_model=value_model, gamma=gamma, lambda_=lambda_, normalize=True
        )
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())


class TestVanilla(unittest.TestCase):
    hidden_sizes = [100]
    num_envs = multiprocessing.cpu_count() - 1
    lr = 1e-2

    def test_cartpole_v0(self):
        env_name = "CartPole-v0"
        env = make_env(env_name, 1)
        (policy, optimizer), _ = create_models(env, self.hidden_sizes, self.lr, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = REINFORCE(policy, optimizer, baseline)
        result = solve(env_name, self.num_envs, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_cartpole_v1(self):
        env_name = "CartPole-v1"
        env = make_env(env_name, 1)
        (policy, optimizer), _ = create_models(env, self.hidden_sizes, self.lr, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = REINFORCE(policy, optimizer, baseline)
        result = solve(env_name, self.num_envs, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_lunar_lander_v2(self):
        env_name = "LunarLander-v2"
        env = make_env(env_name, 1)
        (policy, optimizer), _ = create_models(env, self.hidden_sizes, self.lr, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = REINFORCE(policy, optimizer, baseline)
        result = solve(env_name, env, policy_update, logdir, epochs=500)
        self.assertEqual(result, True)


if __name__ == "__main__":
    unittest.main()
