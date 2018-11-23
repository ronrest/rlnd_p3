import os
import numpy as np
import torch

def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)

def pretty_time(t):
    """ Given a time in seconds, returns a string formatted as "HH:MM:SS" """
    t = int(t)
    H, r = divmod(t, 3600)
    M, S = divmod(r, 60)
    return "{:02n}:{:02n}:{:02n}".format(H,M,S)


def linear_scale_array(x, newmins, newmaxes, oldmins, oldmaxes):
    """ Given an array it linearly scales each of the elements
        independently based on their corresponding old, and new min and max
        values.
    Example:
        >>> linear_scale_array([24, 145],
        >>>                     newmins=[0,-1],
        >>>                     newmaxes=[1, 1],
        >>>                     oldmins=[0,0],
        >>>                     oldmaxes=[100,200])
        array([ 0.24,  0.45])
    """
    # ensure values are numpy arrays
    newmins = np.array(newmins)
    newmaxes = np.array(newmaxes)
    oldmins = np.array(oldmins)
    oldmaxes = np.array(oldmaxes)
    x = np.array(x)

    # TODO: handle oldmaxes and oldmins being Nones
    #       find out min and max from x
    # TODO: handle scalar inputs to mins and maxes, and even x
    ratios = (newmaxes-newmins)/(oldmaxes-oldmins)
    return newmins + ratios*(x-oldmins)


def scale_agent_state(state):
    """ Rescales the states so that they are all within the same -1 to 1 range
    """
    oldmins  = np.tile(np.array([-12, -2, -30, -10, -11, -2, -30, -10], dtype=np.float32), 3)
    oldmaxes = np.tile(np.array([0,    1,  30,  10,  11,  6,  30,  10], dtype=np.float32), 3)
    newmins = -1*np.ones_like(oldmins)
    newmaxes = np.ones_like(oldmaxes)
    newstate = linear_scale_array(state, newmins=newmins, newmaxes=newmaxes, oldmins=oldmins, oldmaxes=oldmaxes)
    return newstate

def tensorfy(x):
    """ Converts an array-like object to a pytorch float tensor """
    return torch.Tensor(x).float()

def tensorfy_experience_samples(samples):
    """ Convert experience samples to tensors and reshape from:
            [n_samples, n_agents, vector_size]
        to
            [n_agents, n_samples, vector_size]
    """
    agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones = zip(*samples)
    agents_states = torch.tensor(agents_states, dtype=torch.float).transpose(0,1)
    global_state = torch.tensor(global_state, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.float).transpose(0,1)
    rewards = torch.tensor(rewards, dtype=torch.float).transpose(0,1).unsqueeze(2)
    next_agents_states = torch.tensor(next_agents_states, dtype=torch.float).transpose(0,1)
    next_global_state = torch.tensor(next_global_state, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float).transpose(0,1).unsqueeze(2)
    return agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones


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
