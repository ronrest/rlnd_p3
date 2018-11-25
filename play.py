#!/usr/bin/env python
# coding: utf-8
import os
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg import MADDPG
from support import tensorfy
import time

# ##############################################################################
#                                                  SETTINGS
# ##############################################################################
MODEL_NAME = "model_g"
ENV_FILE = "Tennis_Linux/Tennis.x86_64"
# ENV_FILE = "Tennis_Linux_NoVis/Tennis.x86_64"

ACTOR_LAYER_SIZES =  [12, 256,128,2]
CRITIC_LAYER_SIZES = [22, 256,128,1]

N_EPISODES = 5    # Number of episodes of game play to observe

SEED = 777
CLAMP_ACTIONS=True

N_AGENTS = 2       # Number of agents in environment
N_ACTIONS = 2      # Number of actions the agent can take
MAX_STEPS = 5000   # max number of steps for an episode

FRAME_DELAY = 0.0  # amount of time to pause between frames

# ##############################################################################
#                                                  SUPPORT
# ##############################################################################
def process_agent_states(states):
    """ Given an array of shape [n_agents, state_size], it
        Only keeps a subset of the state information for each agent """
    # states = scale_agent_state(states)
    return states[:,[3,4,5,11,12,13,16,17,18,19,20,21]]

def process_gobal_state(states):
    """ Given an array of shape [n_agents, state_size], it
        Only keeps a subset of the state information for each agent
        that is not duplicate, or redundant.
        returns an array of shape [18]
    """
    # states = scale_agent_state(states)
    a = states[0,[3,11,16,17,18,19]]
    b = states[1,[3,4,5,11,12,13,16,17,18,19,20,21]]
    c = np.concatenate([a,b])
    return c

# ##############################################################################
#                                                  SETUP
# ##############################################################################
# Generate Output Directory Paths
model_dir = os.path.join("models", MODEL_NAME)
snapshots_dir = os.path.join(model_dir, "snapshots")

# SET SEEDS FOR REPRODUCIBILITY
np.random.seed(SEED)
torch.manual_seed(SEED)


# ##############################################################################
#                                                  ENVIRONMENT
# ##############################################################################
env = UnityEnvironment(file_name=ENV_FILE, seed=SEED)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ##############################################################################
#                                                  AGENT
# ##############################################################################
# INITIALIZE AGENT, AND LOAD WEIGHTS FROM BEST SNAPSHOT
maddpg = MADDPG(actor_layer_sizes=ACTOR_LAYER_SIZES,
                critic_layer_sizes=CRITIC_LAYER_SIZES,
                clamp_actions=CLAMP_ACTIONS,
                logger=None,
                )
maddpg.load_model(os.path.join(snapshots_dir, "best_model.snapshot"))


# ##############################################################################
#                                                 INTERACT WITH ENVIRONMENT
# ##############################################################################
for episode_i in range(1, N_EPISODES+1):
    print("{dec}\nEpisode {i}\n{dec}\n".format(dec="="*60, i=episode_i))
    # INITIALIZE FOR NEW EPISODE
    rewards_this_episode = np.zeros((N_AGENTS,))
    env_info = env.reset(train_mode=False)[brain_name]
    states = process_agent_states(env_info.vector_observations)
    global_state = process_gobal_state(env_info.vector_observations)

    # ITERATE THROUGH EACH STEP OF AN EPISODE
    for step in range(MAX_STEPS):
        # GET ACTIONS TO TAK ADN INTERACT WITH THE ENVIRONMENT
        actions = maddpg.act(tensorfy(states), stacked=True)
        env_info = env.step(actions)[brain_name]

        # EXTRACT AND PROCESS THE RETURNED VALUES FROM ENVIRONMENT
        next_states = process_agent_states(env_info.vector_observations)
        next_global_state = process_gobal_state(env_info.vector_observations)
        rewards = env_info.rewards
        dones = env_info.local_done

        # UPDATE REWARDS
        rewards_this_episode += rewards

        # PREPARE FOR NEXT TIMESTEP
        states = next_states
        global_state = next_global_state

        # END EPISODE IF ANY AGENT IS DONE
        if any(dones):
            print("Episode complete!")
            break

        # PAUSE BETEWEEN FRAMES
        time.sleep(FRAME_DELAY)

    if not any(dones):
        print("Reached max mumber of steps")

    # FEEDBACK PRINTOUT - the score for this episode
    agg_reward_this_episode = np.max(rewards_this_episode)
    print("Episode Score: ", agg_reward_this_episode)

env.close()
