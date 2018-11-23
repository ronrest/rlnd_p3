"""
Contains the neural network to be used by a DDPG agent
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, layer_sizes=[24, 128, 128, 2], actor=False, logger=None, log_layers=False):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(layer_sizes[0],layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1],layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2],layer_sizes[3])
        self.nonlin = f.relu
        self.reset_parameters()
        self.actor = actor
        self.logger = logger
        self.log_layers = log_layers
        self.step = 0

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        h2 = self.nonlin(self.fc2(h1))

        # LOGITS
        # Actor network should return a vector of the actions
        # Critic network should return a q-value number
        if self.actor:
            logits = torch.tanh(self.fc3(h2))
        else:
            logits = self.fc3(h2)

        # MONITOR LAYER VALUES IN TENSORBOARD
        if (self.logger is not None) and (self.log_layers):
            network_name = "actor" if self.actor else "critic"
            self.step += 1
            for key, vals in dict(x=x, h1=h1, h2=h2, logits=logits).items():
                self.logger.add_histogram("{}_layers/{}".format(network_name, key), vals.clone().cpu().data.numpy(), self.step)

        return logits
