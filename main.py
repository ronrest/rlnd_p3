import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from buffer import ReplayBuffer
from maddpg import MADDPG

from support import tensorfy, tensorfy_experience_samples
# Set random seeds
def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ##############################################################################
#                                                  SETTINGS
# ##############################################################################
log_path = os.getcwd()+"/log_a"
model_dir= os.getcwd()+"/model_a"

N_AGENTS = 2           # Number of agents in environment
N_ACTIONS = 2          # Number of actions the agent can take

BUFFER_SIZE = 100000    # Max size of experience replay buffer
BATCH_SIZE = 256        # How many samples to take from buffer with each update
MAX_STEPS = 1000        # max number of steps for an episode
SOLVED_SCORE = 0.5      # Score needed to be considered solved
SOLVED_WINDOW = 100     # Rolling average window size used to evaluate solved score
EPISODES_PER_UPDATE = 1 # How often to update the network

# Amplitude of OU Noise - used for random exploration
# - decayed over time
noise = 1
noise_decay = 1
seed = 1

logger = SummaryWriter(log_dir=log_path)
seeding(seed)

# ##############################################################################
#                                                  SUPPORT
# ##############################################################################
def process_agent_states(states):
    """ Given an array of shape [n_agents, state_size], it
        Only keeps a subset of the state information for each agent """
    # states = scale_agent_state(states)
    return states


def process_gobal_state(states):
    """ Given an array of shape [n_agents, state_size], it
        Only keeps a subset of the state information for each agent
        that is not duplicate, or redundant.
        returns an array of shape [18]
    """
    # states = scale_agent_state(states)
    a = states[0,:]
    b = states[1,:]
    c = np.concatenate([a,b])
    return c

