import time
import os

import numpy as np
import torch
# from tensorboardX import SummaryWriter

from unityagents import UnityEnvironment
from maddpg import MADDPG


from support import scale_agent_state
from support import tensorfy
# from support import maybe_make_dir, pretty_time


UNITY_ENV_FILE = "Tennis_Linux/Tennis.x86_64"
MODEL_NAME = "model_k"
seed = 45

# ##############################################################################
#                                                  SUPPORT
# ##############################################################################
def seeding(seed=1):
    """ Sets random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def process_agent_states(states):
    """ Given an array of shape [n_agents, state_size], it scales
        the features to keep all data within approx the range -1 to 1
    """
    states = scale_agent_state(states)
    return states[:,:]


def process_gobal_state(states):
    """ Given an array of shape [n_agents, state_size], it
        concatenates the agents features to create a flat array
        of shape [stae_size*n_agents]
    """
    states = scale_agent_state(states)
    a = states[0,:]
    b = states[1,:]
    c = np.concatenate([a,b])
    return c


# ##############################################################################
#                                                  SETUP
# ##############################################################################
seeding(seed)


# ##############################################################################
#                                                  ENVIRONMENT
# ##############################################################################
env = UnityEnvironment(file_name=UNITY_ENV_FILE)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations

# calculate dimenstions needed for agent network
agent_state_size = process_agent_states(states).shape[1]
global_state_size = process_gobal_state(states).shape[0]

# ##############################################################################
#                                                  AGENT
# ##############################################################################
N_ACTIONS = 2
N_AGENTS = 2
maddpg = MADDPG(
            actor_layer_sizes=[agent_state_size, 256,256,2],
            critic_layer_sizes=[global_state_size + (N_ACTIONS*N_AGENTS), 256,256,1],
            )

maddpg.load_model(os.path.join("models", MODEL_NAME, "snapshots", "best_model.params"))


# ##############################################################################
#                                                  PLAY WITH ENVIRONMENT
# ##############################################################################
MAX_STEPS = 1000
n_episodes = 1
# t0 = time.time()
frame_delay = 0.01 # amount ot time to pause before moving on to next frame

for episode_i in range(n_episodes):
    # RESET ENVIRONMENT
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations         # get next state (for each agent)
    agents_states = process_agent_states(states)
    global_state = process_gobal_state(states)
    scores = np.zeros(2)
    maddpg.reset_noise()
    # COLLECT A FULL EPISODE OF EXPERIENCES
    # step = 0
    for step in range(MAX_STEPS):
        # GET ACTIONS TO TAKE FROM AGENT
        # Given agent states, get the actions for each agent from the actor policy
        # (and some random noise for exploration)
        # - then convert list of action tensors to a 2D numpy array
        #   [n_agents, n_actions]
        actions = maddpg.act(tensorfy(agents_states))
        actions = torch.stack(actions).detach().numpy()

        # INTERACT WITH THE ENVIRONMENT
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            print("DOne in {} steps".format(step))
            print(scores)
            break

        # PROCESS STATES
        next_agents_states = process_agent_states(next_states)

        # PREPARE FOR NEXT STEP
        # - update state and decay the noise
        agents_states = next_agents_states

        # UPDATE REWARDS
        # rewards_this_episode += rewards

        #print(actions.reshape([1,-1]))
        time.sleep(frame_delay)


env.close()
