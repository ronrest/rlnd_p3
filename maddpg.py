import torch
from ddpg import DDPGAgent
from support import soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def act(self, agents_states, noise=0.0):
        """ Given an array-like object of Tensors, where each tensor represents
            the state of each agent, it passes each of those states to the
            corresponding agent's LOCAL actor network, to get the actions for
            each agent.
        Args:
            agents_states:  array-like object of Tensors
                            [n_agents, state_size]
            noise:          Random noise scaling factor
        Return:
            actions:        Tensor of shape [n_agents, n_actions]
        """
        actions = [agent.act(state, noise) for agent, state in zip(self.agents, agents_states)]
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
            target_actions:        Tensor of shape [n_agents, n_actions]
        """
        target_actions = [agent.target_act(state, noise) for agent, state in zip(self.agents, agents_states)]
        return target_actions

    def update(self, samples, agent_number):
        """
        update the critics and actors of all the agents

        Args:
            samples: Tuple of tensors with the following:
                - agents_states: TODO: XXX shape of tensors[]
                - global_state:
                - actions:
                - rewards:
                - next_agents_states:
                - next_global_state:
                - dones:
            agent_number: is this an integer? for the agent idx?

        """
        agents_states, global_state, actions, rewards, next_agents_states, next_global_state, dones = samples
        agent = self.agents[agent_number]

        # ----------------------------------------------------------------------
        #                               UPDATE CRITIC
        # ----------------------------------------------------------------------
        agent.critic_optimizer.zero_grad()
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_agents_states)
        target_critic_input = torch.cat((next_global_state, *target_actions), dim=-1).to(device)

        #  Observation of the target network
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = rewards[agent_number] + self.discount_factor * q_next * (1 - dones[agent_number])

        # Estimate of future value of local network
        critic_input = torch.cat((global_state, actions[0], actions[1]), dim=-1).to(device)
        q = agent.critic(critic_input)

        # Critic Loss
        critic_loss = self.critic_loss_func(q, y.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ----------------------------------------------------------------------
        #                          UPDATE ACTOR NETWORK
        # ----------------------------------------------------------------------
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # [n_agents, state_size]
        q_input = [self.agents[i].actor(state) if i == agent_number \
                   else self.agents[i].actor(state).detach()
                   for i, state in enumerate(agents_states) ]
        q_input = torch.cat((global_state, *q_input), dim=-1)
        # TODO: what shapes are these taking?

        # get the policy gradient
        actor_loss = -agent.critic(q_input).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # TRACK LOSSES IN TENSORBOARD
        if self.logger is not None:
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            self.logger.add_scalars('loss/agent{}/losses'.format(agent_number),
                           {'critic_loss': cl,
                            'actor_loss': al},
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
