from multiprocessing import Process, Pipe
import numpy as np


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset()
            remote.send(ob)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space, env.spec))
        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True
            )  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space, self.spec = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self, specific_env=None):

        if specific_env is not None:
            self.remotes[specific_env].send(("reset", None))
            return self.remotes[specific_env].recv()

        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
