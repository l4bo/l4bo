from rl_baselines.core import PolicyUpdate, logger, logdir
import multiprocessing


class REINFORCE(PolicyUpdate):
    def update(self, episodes):
        obs, acts, weights = self.batch(episodes)

        dist = self.model(obs)
        log_probs = dist.log_prob(acts)
        loss = -((weights * log_probs).mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"reinforce_loss": loss}


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import (
        solve,
        create_models,
        FullReturnBaseline,
        FutureReturnBaseline,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    logger.info("Using vanilla formulation of policy gradient.")

    hidden_sizes = [100]
    lr = 1e-2
    env, (policy, optimizer), _ = create_models(
        args.env_name, args.num_envs, hidden_sizes, lr, lr
    )
    baseline = FutureReturnBaseline()
    policy_update = REINFORCE(policy, optimizer, baseline)
    solve(
        args.env_name,
        env,
        policy_update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
