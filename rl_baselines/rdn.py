from rl_baselines.core import logger, logdir, gae_advantages
from rl_baselines.ppo import ppo_loss
from rl_baselines.model_updates import ValueUpdate

import multiprocessing
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


class GlobalModel(nn.Module):
    def __init__(self, value_int, value_ext, policy):
        super().__init__()
        self.value_int = value_int
        self.value_ext = value_ext
        self.policy = policy


class IntrinsicValueModel(nn.Module):
    def __init__(self, value_model, random_net, sibling_net):
        super().__init__()
        self.value_model = value_model
        self.random_net = random_net
        for p in self.random_net.parameters():
            p.requires_grad = False
        self.sibling_net = sibling_net

    def forward(self, x):
        return self.value_model(x)

    def intrinsic_rewards(self, obs, obs_mean, obs_std):
        obs = (obs - obs_mean) / (obs_std + 1e-5)
        with torch.no_grad():
            X_r = self.random_net(obs)
        X_r_hat = self.sibling_net(obs)

        rewards_loss = (X_r - X_r_hat) ** 2
        return rewards_loss, X_r


class RDNValueUpdate(ValueUpdate):
    def __init__(
        self,
        value_int,
        value_ext,
        policy,
        optimizer,
        gamma,
        lambda_,
        clip_ratio,
        iters,
        value_int_coeff,
        value_ext_coeff,
        ent_coeff,
    ):
        nn.Module.__init__(self)
        self.value_int = value_int
        self.value_ext = value_ext
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.iters = iters
        self.value_int_coeff = value_int_coeff
        self.value_ext_coeff = value_ext_coeff
        self.ent_coeff = ent_coeff

    def update(self, episodes):
        # Remove last observation, it's not needed for the update
        obs = episodes.obs
        acts = episodes.acts

        B, N, *obs_shape = obs.shape

        batch_mean = obs.view(B * N, *obs_shape).mean(dim=0).clone().detach()
        batch_var = obs.view(B * N, *obs_shape).var(dim=0).clone().detach()
        batch_count = B * N
        with torch.no_grad():
            if not hasattr(self, "obs_mean"):
                self.obs_mean = batch_mean
                self.obs_var = batch_var
                self.obs_count = batch_count
            else:
                tot_count = batch_count + self.obs_count
                delta = batch_mean - self.obs_mean
                self.obs_mean += delta * batch_count / tot_count

                self.obs_count += batch_count
                m_a = self.obs_var * self.obs_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + (delta) ** 2 * self.obs_count * batch_count / tot_count
                self.obs_var = M2 / tot_count

            obs_std = torch.sqrt(self.obs_var)
            rews_int, X_r = self.value_int.intrinsic_rewards(
                obs, self.obs_mean, obs_std
            )

            current_obs = obs[:, :-1, ...]
            next_obs = obs[:, -1, ...]
            next_ext = self.value_ext(next_obs)
            next_int = self.value_int(next_obs)

            values_int = self.value_int(obs)
            values_ext = self.value_ext(obs)

            # Intrinsic reward is non-episodic.
            dones = torch.zeros(*episodes.dones.shape)
            advantages = torch.zeros(*episodes.rews.shape)
            adv_int = gae_advantages(
                advantages, values_int, dones, rews_int, self.gamma, self.lambda_
            )
            adv_ext = episodes.gae_advantages(values_ext, self.gamma, self.lambda_)

            advs = adv_int * self.value_int_coeff + adv_ext * self.value_ext_coeff

            old_dist = self.policy(current_obs)
            old_log_probs = old_dist.log_prob(acts)

        returns_ext = episodes.discounted_returns(
            gamma=self.gamma, pred_values=next_ext
        )
        returns_int = episodes.discounted_returns(
            gamma=self.gamma, pred_values=next_int
        )
        for i in range(self.iters):
            pred_values_ext = self.value_ext(current_obs)
            loss_ext = F.mse_loss(pred_values_ext, returns_ext)

            pred_values_int = self.value_int(current_obs)
            loss_int = F.mse_loss(pred_values_int, returns_int)

            # No coeff here, we try to learn value functions, not policy
            value_loss = loss_ext + loss_int

            rews_int, X_r = self.value_int.intrinsic_rewards(
                obs, self.obs_mean, obs_std
            )

            pi_loss, kl, clipfrac, entropy = ppo_loss(
                self.policy, current_obs, acts, advs, old_log_probs, self.clip_ratio
            )

            aux_loss = rews_int.mean()

            loss = pi_loss + value_loss + self.ent_coeff * entropy + aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            total_norm = 0
            for p in self.policy.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            for p in self.value_int.parameters():
                if p.requires_grad:
                    try:
                        param_norm = p.grad.data.norm(2)
                    except Exception:
                        import ipdb

                        ipdb.set_trace()
                    total_norm += param_norm.item() ** 2
            for p in self.value_ext.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            self.optimizer.step()
            losses = {
                "value_ext": loss_ext,
                "value_int": loss_int,
                "pi_loss": pi_loss,
                "ent": entropy,
                "clipfrac": clipfrac,
                "approx_kl": kl,
                "aux_loss": aux_loss,
                "featvar": X_r.var(),
                "featmax": torch.abs(X_r).max(),
                "grad_norm": total_norm,
            }

            if i == 0:
                logger.debug("\t".join(f"{ln:>12}" for ln in losses))
            logger.debug("\t".join(f"{l:12.4f}" for l in losses.values()))

        return losses


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, default_policy_model, make_env, default_model
    from rl_baselines.baselines import DiscountedReturnBaseline, GAEBaseline
    from rl_baselines.model_updates import ActorCriticUpdate, ValueUpdate
    from rl_baselines.models import ValueModel

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", "--env", type=str, default="PitfallNoFrameskip-v4"
    )
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--value-ext-coeff", type=float, default=2)
    parser.add_argument("--value-int-coeff", type=float, default=1)
    parser.add_argument("--ent-coeff", type=float, default=1e-3)
    parser.add_argument("--target-kl", type=float, default=0.01)

    # 128 steps * 32 env in OpenAI
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    logger.info("Using PPO formulation of policy gradient.")

    hidden_sizes = [100]
    env = make_env(args.env_name, args.num_envs)
    policy = default_policy_model(env, hidden_sizes)

    value_int_model = default_model(env, hidden_sizes, 1)
    random_net = default_model(env, [90], 1)
    sibling_net = default_model(env, [80], 1)

    value_int = IntrinsicValueModel(value_int_model, random_net, sibling_net)
    value_ext = ValueModel(default_model(env, hidden_sizes, 1))
    global_model = GlobalModel(value_int, value_ext, policy)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    update = RDNValueUpdate(
        value_int,
        value_ext,
        policy,
        optimizer,
        args.gamma,
        args.lam,
        args.clip_ratio,
        args.iters,
        args.value_int_coeff,
        args.value_ext_coeff,
        args.ent_coeff,
    )
    solve(
        args.env_name,
        env,
        update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
