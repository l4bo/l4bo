import torch


class Baseline:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _get(self, episodes):
        raise NotImplementedError

    def __call__(self, episodes):
        batch_weights = self._get(episodes)
        weights = torch.Tensor(batch_weights)
        if self.normalize:
            weights = (weights - weights.mean()) / (weights.std() + 1e-5)
        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"


class FullReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = [episode.ret for episode in episodes for _ in episode.rew]
        return weights


class FutureReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = []
        for episode in episodes:
            ret = 0
            returns = []
            for rew in reversed(episode.rew):
                ret += rew
                returns.append(ret)
            weights += list(reversed(returns))
        return weights


def discounted_cumsum(elements, gamma):
    discounted = []
    cur = 0
    for element in reversed(elements):
        cur = element + gamma * cur
        discounted.append(cur)
    return list(reversed(discounted))


class DiscountedReturnBaseline(Baseline):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def _get(self, episodes):
        weights = []
        for episode in episodes:
            weights += discounted_cumsum(episode.rew, self.gamma)
        return weights


class GAEBaseline(Baseline):
    def __init__(self, value_model, gamma, lambda_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value_model = value_model
        assert gamma > lambda_, f"For stability λ({lambda_}) < γ({gamma}) is advised"

    def _get(self, episodes):
        obs = [item for episode in episodes for item in episode.obs]
        with torch.no_grad():
            values = self.value_model(torch.Tensor(obs))
        start = 0
        weights = []
        for episode in episodes:
            end = start + len(episode.obs)
            # End value always 0 as we never cut episodes.
            v_pi = torch.cat((values[start:end], torch.Tensor([[0]])), dim=0)
            deltas = (
                torch.Tensor(episode.rew).unsqueeze(1)
                + self.gamma * v_pi[1:]
                - v_pi[:-1]
            )
            assert list(deltas.shape) == [len(episode.obs), 1], "Dimension sanity check"
            weights += discounted_cumsum(deltas, self.gamma * self.lambda_)
            start = end
        return torch.Tensor(weights)
