from rl_baselines.core import logger, logdir
from rl_baselines.ppo import PPO
import multiprocessing
import torch


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, create_models, make_env
    from rl_baselines.baselines import DiscountedReturnBaseline, GAEBaseline
    from rl_baselines.model_updates import ActorCriticUpdate, ValueUpdate

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
    policy_update = PPO(
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
