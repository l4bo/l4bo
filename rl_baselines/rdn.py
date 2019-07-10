from rl_baselines.core import logger, logdir
from rl_baselines.ppo import PPO
import multiprocessing
import torch


class RDN(PPO):
    def loss(self, policy, episodes, obs, acts, weights, old_log_probs):
        clip_ratio = self.clip_ratio
        dist = policy(obs)
        log_probs = dist.log_prob(acts)

        diff = log_probs - old_log_probs
        ratio = (diff).exp()
        approx_kl = (-diff).mean().item()
        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        clipfrac = (
            ((ratio > (1 + clip_ratio)) | (ratio < (1 - clip_ratio))).float().mean()
        )

        loss = -(torch.min(ratio * weights, clipped * weights)).mean()
        entropy = dist.entropy().mean()
        return loss, approx_kl, clipfrac, entropy

    def update(self, episodes):
        obs, acts, weights = self.batch(episodes)

        with torch.no_grad():
            old_dist = self.policy(obs)
            old_log_probs = old_dist.log_prob(acts)
        # logger.debug("\t".join(["pi_loss", "kl", "clipfrac", "entropy"]))
        for i in range(self.policy_iters):
            self.optimizer.zero_grad()
            loss, kl, clipfrac, entropy = self.loss(
                self.policy, episodes, obs, acts, weights, old_log_probs
            )
            if kl > self.target_kl:
                logger.warning(
                    f"Stopping after {i} iters because KL > {self.target_kl}"
                )
                break
            # logger.debug("\t".join("%.4f" % v for v in [loss, kl, clipfrac, entropy]))
            loss.backward()
            self.optimizer.step()
        return {"ppo_loss": loss}


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import (
        solve,
        create_models,
        make_env,
        GAEBaseline,
        ActorCriticUpdate,
        ValueUpdate,
        DiscountedReturnBaseline,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", "--env", type=str, default="PitfallNoFrameskip-v4"
    )
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.2)
    parser.add_argument("--policy-iters", type=int, default=80)
    parser.add_argument("--value-iters", type=int, default=80)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info("Using PPO formulation of policy gradient.")

    hidden_sizes = [100]
    env = make_env(args.env_name, args.num_envs)
    (policy, optimizer), (value, vopt) = create_models(
        env, hidden_sizes, args.pi_lr, args.vf_lr
    )

    baseline = GAEBaseline(value, gamma=args.gamma, lambda_=args.lam)
    policy_update = RDN(
        args.policy_iters, args.clip_ratio, args.target_kl, policy, optimizer, baseline
    )

    vbaseline = DiscountedReturnBaseline(gamma=args.gamma, normalize=False)
    value_update = ValueUpdate(value, vopt, vbaseline, iters=args.value_iters)
    update = ActorCriticUpdate(policy_update, value_update)
    solve(
        args.env_name,
        env,
        update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
