from rl_baselines.core import logger, logdir, gae_advantages
from rl_baselines.ppo import ppo_loss
from rl_baselines.model_updates import ValueUpdate

import multiprocessing
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.value_int = value_int
        # self.value_ext = value_ext
        # self.policy = policy

        self.convs = nn.ModuleList()
        layers = [
            {"out_channels": 32, "kernel_size": 8, "stride": 4},
            {"out_channels": 64, "kernel_size": 4, "stride": 2},
            {"out_channels": 64, "kernel_size": 4, "stride": 1},
        ]
        in_channels = 4
        for layer in layers:
            self.convs.append(nn.Conv2d(in_channels=in_channels, **layer))
            in_channels = layer["out_channels"]

        self.activation = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 448)

    def forward(self, obs):
        assert len(obs.shape) == 5
        E, B, H, W, C = obs.shape
        assert C == 4  # Frame stacks
        # Fuse environments and steps dimensions
        x = obs.contiguous().view(-1, H, W, C)
        # Pytorch uses C, H, W for its convolution
        x = x.permute(0, 3, 1, 2)
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)

        x = self.activation(self.fc1(x))

        return x, x


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
        model,
        optimizer,
        gamma,
        lambda_,
        clip_ratio,
        iters,
        value_int_coeff,
        value_ext_coeff,
        ent_coeff,
        num_mini_batches,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.iters = iters
        self.value_int_coeff = value_int_coeff
        self.value_ext_coeff = value_ext_coeff
        self.ent_coeff = ent_coeff
        self.num_mini_batches = num_mini_batches

    def policy(self, obs):
        return self.model.policy(obs)

    def loss(
        self,
        returns_int,
        returns_ext,
        acts,
        advs,
        old_log_probs,
        obs,
        obs_mean,
        obs_std,
    ):
        current_obs = obs[:, :-1, ...]
        pred_values_int = self.value_int(current_obs)
        loss_int = F.mse_loss(pred_values_int, returns_int)

        pred_values_ext = self.value_ext(current_obs)
        loss_ext = F.mse_loss(pred_values_ext, returns_ext)

        # No coeff here, we try to learn value functions, not policy
        value_loss = loss_ext + loss_int

        rews_int, X_r = self.value_int.intrinsic_rewards(obs, self.obs_mean, obs_std)

        pi_loss, kl, clipfrac, entropy = ppo_loss(
            self.policy, current_obs, acts, advs, old_log_probs, self.clip_ratio
        )

        aux_loss = rews_int.mean()

        loss = pi_loss + value_loss + self.ent_coeff * entropy + aux_loss
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
        }
        return loss, losses

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
        nperbatch = B // self.num_mini_batches
        print(nperbatch)
        for i in range(self.iters):
            for start in range(0, B, nperbatch):
                end = start + nperbatch
                slice_ = slice(start, end)

                print(i, start)

                loss, losses = self.loss(
                    returns_int[slice_],
                    returns_ext[slice_],
                    acts[slice_],
                    advs[slice_],
                    old_log_probs[slice_],
                    obs[slice_],
                    self.obs_mean,
                    obs_std,
                )

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
                losses["grad_norm"] = total_norm
                self.optimizer.step()

                if i == 0 and start == 0:
                    logger.debug("\t".join(f"{ln:>12}" for ln in losses))
                logger.debug("\t".join(f"{l:12.4f}" for l in losses.values()))

        return losses


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, default_policy_model, make_env, default_model
    from rl_baselines.models import ValueModel

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", "--env", type=str, default="PitfallNoFrameskip-v4"
    )
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--value-ext-coeff", type=float, default=2)
    parser.add_argument("--value-int-coeff", type=float, default=1)
    parser.add_argument("--ent-coeff", type=float, default=1e-3)
    parser.add_argument("--target-kl", type=float, default=0.01)

    # 128 steps * 32 env in OpenAI
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--num_mini_batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    logger.info("Using PPO formulation of policy gradient.")

    model = GlobalModel()
    env = make_env(args.env_name, args.num_envs, frame_stack=args.frame_stack)

    # policy = default_policy_model(env, hidden_sizes)
    # value_int_model = default_model(env, hidden_sizes, 1)
    # random_net = default_model(env, hidden_sizes, 1)
    # sibling_net = default_model(env, hidden_sizes, 1)
    # value_int = IntrinsicValueModel(value_int_model, random_net, sibling_net)
    # value_ext = ValueModel(default_model(env, hidden_sizes, 1))

    global_model = GlobalModel()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    assert (
        args.num_envs > args.num_mini_batches
    ), "We need more environments than minibatches."

    update = RDNValueUpdate(
        global_model,
        optimizer,
        args.gamma,
        args.lam,
        args.clip_ratio,
        args.iters,
        args.value_int_coeff,
        args.value_ext_coeff,
        args.ent_coeff,
        args.num_mini_batches,
    )
    solve(
        args.env_name,
        env,
        update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
