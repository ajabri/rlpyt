
import numpy as np
import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel, MultiFfModel

# MIN_STD = 1e-6

class MultiAgentGaussianPgAgent(BaseAgent):
    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value = self.model(*model_inputs)

        samples = (DistInfoStd(mean=mu, log_std=log_std), value)
        return buffer_to(samples, device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)

        for _a_space in env_spaces.action.space:
            assert len(_a_space.shape) == 1
            assert len(np.unique(_a_space.high)) == 1
            assert np.all(_a_space.low == -_a_space.high)
            
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[-1],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value = self.model(*model_inputs)
        # import pdb; pdb.set_trace()

        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, value = self.model(*model_inputs)
        return value.to("cpu")

class MultiFfAgent(MultiAgentGaussianPgAgent):

    def __init__(self, ModelCls=MultiFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        # assert len(env_spaces.action.shape) == 1
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.shape)

