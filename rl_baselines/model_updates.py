import torch.nn as nn
import torch.nn.functional as F
import torch
from rl_baselines.baselines import DiscountedReturnBaseline


class ModelUpdate(nn.Module):
    loss_name = "loss"

    def __init__(self, model, optimizer, baseline, iters=1):
        super().__init__()
        self.baseline = baseline
        self.model = model
        self.optimizer = optimizer
        self.iters = iters

    def update(self, episodes):
        raise NotImplementedError

    def batch(self, episodes):
        batch_obs = [item for episode in episodes for item in episode.obs]
        batch_acts = [item for episode in episodes for item in episode.act]
        weights = self.baseline(episodes)
        obs = torch.Tensor(batch_obs)
        acts = torch.stack(batch_acts, dim=0)
        return obs, acts, weights


class PolicyUpdate(ModelUpdate):
    loss_name = "policy_loss"

    @property
    def policy(self):
        return self.model


class ValueUpdate(ModelUpdate):
    loss_name = "value_loss"

    @property
    def value(self):
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.baseline, DiscountedReturnBaseline
        ), "Value models need to learn discounted returns"

    def update(self, episodes):
        obs, _, returns = self.batch(episodes)

        for i in range(self.iters):
            values = self.model(obs)
            loss = F.mse_loss(values, returns.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {self.loss_name: loss}


class ActorCriticUpdate(nn.Module):
    def __init__(self, policy_update, value_update):
        super().__init__()
        assert isinstance(policy_update, PolicyUpdate)
        assert isinstance(value_update, ValueUpdate)
        self.policy_update = policy_update
        self.value_update = value_update

    @property
    def policy(self):
        return self.policy_update.policy

    @property
    def value(self):
        return self.value_update.value

    def update(self, episodes):
        losses = {}
        p_losses = self.policy_update.update(episodes)
        v_losses = self.value_update.update(episodes)
        losses.update(p_losses)
        losses.update(v_losses)
        return losses