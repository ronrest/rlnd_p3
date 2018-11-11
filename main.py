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

# ##############################################################################
#                                                  ENVIRONMENT
# ##############################################################################
env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64")
# env = UnityEnvironment(file_name="../Tennis_Linux_NoVis/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
action_type = brain.vector_action_space_type

states = env_info.vector_observations
#state_size = brain.vector_observation_space_size # gives wrong value for this environment
state_size = states.shape[1]
state_type = brain.vector_observation_space_type

print('Number of agents               :', num_agents)
print('Action Shape                   :', action_size)
print("Action Type                    :", action_type)
print("State Shape (all agents)       :", states.shape)
print("State Shape (individual agent) :", state_size)
print("State Type                     :", state_type)
print('\nExample state for a single agent:\n', states[0])

agent_state_size = process_agent_states(states).shape[1]
global_state_size = process_gobal_state(states).shape[0]


# ##############################################################################
#                                                  AGENT
# ##############################################################################
# initialize policy and critic
maddpg = MADDPG(
            actor_layer_sizes=[agent_state_size, 128,128,2],
            critic_layer_sizes=[global_state_size + (N_ACTIONS*N_AGENTS), 128,128,1],
            lr_actor=1e-3,
            lr_critic=1e-3,
            discount_factor=0.95,
            logger=logger,
            )


