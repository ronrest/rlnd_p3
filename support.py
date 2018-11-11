import numpy as np
import torch

def tensorfy(x):
    """ Converts an array-like object to a pytorch float tensor """
    return torch.Tensor(x).float()

def random_swap(x):
    """ Randomly swaps the first two rows of data in an array-like object
    """
    if np.random.randint(2) == 1:
        return x[[1,0]]
    else:
        return x

def random_swap_agent_experiences(experiences):
    """ Randomly swaps the experiences of agent1/agent2
    """
    (agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones) = experiences
    agents_states = random_swap(agents_states)
    actions = random_swap(actions)
    rewards = random_swap(rewards)
    next_agents_states = random_swap(next_agents_states)
    dones = random_swap(dones)
    return (agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
