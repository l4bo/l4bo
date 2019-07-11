## L4BO

### Projects

- **energy_rl**: *Study of back-propagation in energy constrained environments*
  - PPO: `energy_reinforce_train energy/configs/config.json --device cpu`
  - ES: `energy_es_train energy/configs/config.json --device cpu`



Testing: `python -m rl_baselines.reinforce --env-name="CartPole-v0`

Checking the training results (in another shell): `tensorboard --logdir=runs/Jul08_XX-XX-XX/` (requires pytorch > 1.1 & tensorboard > 1.14 (`pip install tb-nightly`)

Running an agent to see how it operates: `python -m rl_baselines.test_agent --model=run/Jul08_XX-XX-XX/checkpoint.pth --env-name=CartPole-v0`
