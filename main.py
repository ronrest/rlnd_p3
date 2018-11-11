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


# ADD GRAPH TO TENSORBOARD
#actor_input_dummy = torch.autograd.Variable(torch.rand(1, agent_state_size))
#logger.add_graph(maddpg.agents[0].actor, (actor_input_dummy, ), verbose=True)
critic_input_dummy = torch.autograd.Variable(torch.rand(1, global_state_size + (N_ACTIONS*N_AGENTS)))
logger.add_graph(maddpg.agents[0].critic, (critic_input_dummy, ), verbose=False)


# ##############################################################################
#                                                  TRAIN LOOP
# ##############################################################################
buffer = ReplayBuffer(BUFFER_SIZE)

n_episodes = 5000
solved_printout = False
best_rolling_mean_reward = -np.inf
history = []            # history of actual reward at each episode
history_rolling = []    # history of rolling mean rewards over time

for episode_i in range(1, n_episodes+1):
    rewards_this_episode = np.zeros(N_AGENTS, dtype=np.float)

    # RESET ENVIRONMENT
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations         # get next state (for each agent)
    agents_states = process_agent_states(states)
    global_state = process_gobal_state(states)
    scores = np.zeros(2)

    # COLLECT A FULL EPISODE OF EXPERIENCES
    for step in range(MAX_STEPS):
        # RESET OUNOISE
        # maddpg.reset_noise()
        for i in range(N_AGENTS):
            maddpg.agents[i].noise.reset()
        # PERFORM SINGLE STEP

        # GET ACTIONS TO TAKE FROM AGENT
        # Given agent states, get the actions for each agent from the actor policy
        # (and some random noise for exploration)
        # - then convert list of action tensors to a 2D numpy array
        #   [n_agents, n_actions]
        actions = maddpg.act(tensorfy(agents_states), noise=noise)
        actions = torch.stack(actions).detach().numpy()

        # INTERACT WITH THE ENVIRONMENT
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
           break
           print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

        # PROCESS STATES
        next_agents_states = process_agent_states(next_states)
        next_global_state = process_gobal_state(next_states)

        # ADD EXPERIENCE TO REPLAY BUFFER
        experience = (agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones)
        buffer.push(experience)

        # UPDATE REWARDS
        rewards_this_episode += rewards

        # PREPARE FOR NEXT STEP
        # - update state and decay the noise
        #states = next_states           # roll over states to next time step
        agents_states = next_agents_states
        global_state = next_global_state
        noise *= noise_decay

    # UPDATE NETWORK - once after every EPISODES_PER_UPDATE
    if (len(buffer) > BATCH_SIZE) and ((episode_i % EPISODES_PER_UPDATE) == 0):
        for _ in range(5):
            for agent_i in range(N_AGENTS):
                samples = buffer.sample(BATCH_SIZE)
                samples = tensorfy_experience_samples(samples)
                samples = random_swap_agent_experiences(samples)
                maddpg.update(samples, agent_i)
        maddpg.update_targets() #soft update the target network towards the actual networks

    # UPDATE REWARDS
    agg_reward_this_episode = rewards_this_episode.max()
    # agg_reward_this_episode = rewards_this_episode.mean()
    rolling_mean_reward = np.mean(history[-SOLVED_WINDOW:])
    history.append(agg_reward_this_episode)
    history_rolling.append(rolling_mean_reward)

    # FEEDBACK
    acc = str([list(actions[0]), list(actions[1])] )
    agent_rewards_string = ["{: 3.3f}".format(r) for r in rewards_this_episode]
    agent_rewards_string = ",".join(agent_rewards_string)
    feedback = "\r{ep} Rolling Mean Reward: {rm: 3.3f}  Avg Reward This episode: {re: 3.3f}  Individual rewards [{ar}] acc: {acc}".format(ep=episode_i, rm=rolling_mean_reward, re=agg_reward_this_episode, ar=agent_rewards_string, acc=acc)
    print(feedback, end="")

    # LIVE PLOTS
    logger.add_scalars('rewards/Rewards_this_episode',
                   {'Agent_0': rewards_this_episode[0],
                    'Agent_1': rewards_this_episode[1]},
                   episode_i)
    logger.add_scalars('rewards/Aggregated_Rewards_Over_Time',
                   {
                    'Rolling mean reward': rolling_mean_reward,
                    'Reward this episode (max agent)': agg_reward_this_episode,
                    },
                   episode_i)
    logger.add_scalars('actions/Actions_Agent_0',
                   {'Action_0': actions[0][0],
                    'Action_1': actions[0][1]},
                   episode_i)
    logger.add_scalars('actions/Actions_Agent_1',
                   {'Action_0': actions[1][0],
                    'Action_1': actions[1][1]},
                   episode_i)
    logger.add_scalars('noise/noise', {"noise": noise}, episode_i)

    for name, param in maddpg.agents[0].critic.named_parameters():
        logger.add_histogram("critic_weights/{}".format(name), param.clone().cpu().data.numpy(), episode_i)
    for name, param in maddpg.agents[0].actor.named_parameters():
        logger.add_histogram("actor_weights/{}".format(name), param.clone().cpu().data.numpy(), episode_i)


