import numpy as np
import torch
from network import Network
from support import hard_update
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


class DDPGAgent:
    def __init__(self, actor_layer_sizes=[24, 128,128,2], critic_layer_sizes=[24, 128,128,1], lr_actor=1.0e-2, lr_critic=1.0e-2, clamp_actions=True, logger=None):
        super(DDPGAgent, self).__init__()

        # SET UP ACTOR AND CRITIC NETWORKS
        self.actor = Network(layer_sizes=actor_layer_sizes, actor=True, logger=logger).to(device)
        self.critic = Network(layer_sizes=critic_layer_sizes, logger=logger).to(device)
        self.target_actor = Network(layer_sizes=actor_layer_sizes, actor=True, logger=logger).to(device)
        self.target_critic = Network(layer_sizes=critic_layer_sizes, logger=logger).to(device)

        # INITIALIZE TARGET NETWORKS TO HAVE SAME WEIGHTS AS LOCAL NETWORKS
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # OPTIMIZERS
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0.0)

        # NOISE - for exploration of actions
        self.noise = OUNoise(actor_layer_sizes[-1], scale=1.0 )
        self.clamp_actions = clamp_actions


    def act(self, obs, noise=0.0):
        """ Given a tensor representing the states, it returns the predicted
            actions the agent should take using the LOCAL network.

            If `noise` is provided, it adds some random noise to the actions
            to make the agent explore.
        """
        obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        if self.clamp_actions:
            action = torch.clamp(action, -1.0, 1.0)
        return action

    def target_act(self, obs, noise=0.0):
        obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        if self.clamp_actions:
            action = torch.clamp(action, -1.0, 1.0)
        return action
