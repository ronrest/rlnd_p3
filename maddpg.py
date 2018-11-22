"""
Implements the MADDPG algoritm, and supporting functions for this algorithm.

Credits:
--------
The MADDPG class is a modified version of code originally sourced from a
Udacity Lab for the Deep Reinforcement Learning Nanodegree
"""
import torch
from ddpg import DDPGAgent
from support import soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ##############################################################################
#                                  SUPPORT
# ##############################################################################
def tensorfy_experience_samples(samples):
    """ Convert experience samples from:
            (n_samples, n_experience_items, Array[n_agents, vector_size])
        to
            (n_experience_items, Tensor[n_agents, n_samples, vector_size])

        the experience items should be the following:
        - agents_states
        - global_state
        - actions
        - rewards
        - next_agents_states
        - next_global_state
            - dones
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


class MADDPG(object):
    def __init__(self, actor_layer_sizes=[24, 128,128,2], critic_layer_sizes=[52, 128,128,1], discount_factor=0.95, tau=0.02, logger=None, lr_actor=1.0e-2,  lr_critic=1.0e-2):
        super(MADDPG, self).__init__()
        self.agents = [DDPGAgent(actor_layer_sizes=actor_layer_sizes,
                                critic_layer_sizes=critic_layer_sizes,
                                lr_actor=lr_actor,
                                lr_critic=lr_critic,
                                logger=logger,
                                ),
                            DDPGAgent(
                                actor_layer_sizes=actor_layer_sizes,
                                critic_layer_sizes=critic_layer_sizes,
                                lr_actor=lr_actor,
                                lr_critic=lr_critic,
                                logger=logger,
                                ),
                            ]
        self.n_agents = len(self.agents)
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.logger = logger
        self.critic_loss_func = torch.nn.SmoothL1Loss()


    def get_actors(self):
        """Return the actors of all the agents in the MADDPG object"""
        actors = [agent.actor for agent in self.agents]
        return actors

    def get_target_actors(self):
        """Return the target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.target_actor for agent in self.agents]
        return target_actors

    def act(self, agents_states, noise=0.0, stacked=False):
        """ Given an array-like object of Tensors, where each tensor represents
            the state of each agent, it passes each of those states to the
            corresponding agent's LOCAL actor network, to get the actions for
            each agent.
        Args:
            agents_states:  array-like object of Tensors
                            [n_agents, state_size]
            noise:          Random noise scaling factor
        Return:
            actions:        Numpy array of shape [n_agents, n_actions]
        """
        actions = [agent.act(state, noise=noise) for agent, state in zip(self.agents, agents_states)]
        if stacked:
            actions = np.vstack(actions[i].detach().numpy() for i in range(self.n_agents))
        return actions

    def target_act(self, agents_states, noise=0.0):
        """ Given an array-like object of Tensors, where each tensor represents
            the state of each agent, it passes each of those states to the
            corresponding agent's TARGET actor network, to get the actions for
            each agent.
        Args:
            agents_states:  array-like object of Tensors
                            [n_agents, state_size]
            noise:          Random noise scaling factor
        Return:
            actions:        Numpy array of shape [n_agents, n_actions]
        """
        actions = [agent.target_act(state, noise) for agent, state in zip(self.agents, agents_states)]
        # actions = np.vstack(actions[i].detach().numpy() for i in range(self.n_agents))
        return actions

    def update(self, samples, agent_number):
        """ Given samples from the experience replay buffer in the shape
                (n_samples, n_experience_items, Array[n_agents, vector_size])
            then it updates the actor and critic networks for each agent.
        """
        # PREPROCESS THE EXPERIENCE SAMPLES
        # Convert the arrays in the experience tuple from:
        #     [n_samples, n_agents, vector_size]
        # to:
        #     [n_agents, n_samples, vector_size]
        states, global_state, actions, rewards, next_states, next_global_state, dones = tensorfy_experience_samples(samples)

        # ----------------------------------------------------------------------
        #                         UPDATE CRITIC NETWORKS
        # ----------------------------------------------------------------------
        agent = self.agents[agent_number]
        agent.critic_optimizer.zero_grad()

        # ESTIMATE OF FUTURE REWARDS
        # - Using global state instead of individual agent state to make
        #   training more stable
        # - q =  Q(global_state)
        critic_input = torch.cat((global_state, *actions), dim=1).to(device)
        q = agent.critic(critic_input)

        # PROXY FOR ACTUAL FUTURE RETURNS
        # - Calculated as actual observation of current rewards + discounted
        #   returns of estimate of future returns from next state using the
        #   more stable target network
        # - y = current reward + discount * Q'(st+1,at+1)
        target_actions = self.target_act(next_states)
        target_critic_input = torch.cat((next_global_state, *target_actions), dim=1).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))

        # CRITIC LOSS
        # - Using difference between estimated q, and y
        #critic_loss_func = torch.nn.MSELoss()
        critic_loss_func = torch.nn.SmoothL1Loss()
        critic_loss = critic_loss_func(q, y.detach())

        # GRADIENTS AND UPDATE - potentially also clip gradients
        critic_loss.backward()
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.gradient_clipping)
        agent.critic_optimizer.step()

        # ----------------------------------------------------------------------
        #                          UPDATE ACTOR NETWORKS
        # ----------------------------------------------------------------------
        agent.actor_optimizer.zero_grad()

        # INPUT FOR AGENT ACTOR NETWORK
        # - Combines all states of all agents, and all actions from local
        #   network
        # - detach the other agents to  save time in computing derivative
        actor_actions = [self.agents[i].actor(state) if i == agent_number \
                        else self.agents[i].actor(state).detach() \
                        for i, state in enumerate(states) ]
        q_input = torch.cat((global_state, *actor_actions), dim=1)

        # ACTOR LOSS
        actor_loss = -agent.critic(q_input).mean()

        # GRADIENTS AND UPDATE - potentially also clip gradients
        actor_loss.backward()
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),self.gradient_clipping)
        agent.actor_optimizer.step()

        # MONITOR LOSS IN TENSORBOARD
        if self.logger is not None:
            self.logger.add_scalars('losses/agent{}'.format(agent_number),
                            {
                            'critic loss': critic_loss.cpu().detach().item(),
                            'actor_loss': actor_loss.cpu().detach().item(),
                            },
                           self.iter)


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for agent in self.agents:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)

    def reset_noise(self):
        for agent in self.agents:
            agent.noise.reset()

    def save_model(self, filename):
        save_dict_list =[]
        for i in range(self.n_agents):
            save_dict = {
                'actor_params' : self.agents[i].actor.state_dict(),
                'actor_optim_params': self.agents[i].actor_optimizer.state_dict(),
                'critic_params' : self.agents[i].critic.state_dict(),
                'critic_optim_params' : self.agents[i].critic_optimizer.state_dict()
                }
            save_dict_list.append(save_dict)
        torch.save(save_dict_list, filename)

    def load_model(self, filename):
        params = torch.load(filename)
        for i in range(self.n_agents):
            self.agents[i].actor.load_state_dict(params[i]['actor_params'])
            self.agents[i].actor_optimizer.load_state_dict(params[i]['actor_optim_params'])
            self.agents[i].critic.load_state_dict(params[i]['critic_params'])
            self.agents[i].critic_optimizer.load_state_dict(params[i]['critic_optim_params'])
