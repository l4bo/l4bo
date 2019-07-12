from rl_baselines.environment import (
    SubprocVecEnv,
    make_atari,
    ScaledFloatFrame,
    ClipRewardEnv,
    FrameStack,
)
import multiprocessing
import torch
import numpy as np
import tqdm
import gym
import cv2


# Overload to 64x64
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 64
        self.height = 64
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def observation(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height))
        return frame
        return frame[:, :, None]


def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    # env = NormalizeObservation(env)
    return env


def main():
    num_envs = multiprocessing.cpu_count() - 1
    env_name = "CarRacing-v0"
    env = SubprocVecEnv(
        [
            # lambda: wrap_deepmind(make_atari(env_name), frame_stack=True)
            lambda: wrap_deepmind(gym.make(env_name))
            for i in range(num_envs)
        ]
    )
    obs = env.reset()

    H, W, T = env.observation_space.shape

    total = 5000
    B = total // num_envs
    observations = torch.zeros((B, num_envs, H, W, T))

    for i in tqdm.tqdm(range(B)):
        observations[i] = torch.from_numpy(obs).float()
        acts = np.array([env.action_space.sample() for j in range(num_envs)])
        obs, _, _, _ = env.step(acts)

    out = observations.reshape(-1, H, W, T)
    out = out.permute(0, 3, 1, 2)
    out = out / 255.0
    import ipdb

    ipdb.set_trace()
    torch.save(out, "frames.pty")


if __name__ == "__main__":
    main()
