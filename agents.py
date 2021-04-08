from collections import defaultdict
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.action_selector import ActionSelector
from core.buffer_indexer import BufferIndexer
from core.eps_strategy import EpsilonStrategy
from core.rewards import *


class Agent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 env=gym.make('MountainCar-v0'),
                 lr=0.0003,
                 gamma=0.99,
                 update_model_frequency=3,
                 buffer_indexer=BufferIndexer(),
                 eps_strategy=EpsilonStrategy(decay=5e-3),
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        # Environment
        if len(state_dim) == 1:
            self.state_dim = state_dim[0]
        else:
            self.state_dim = state_dim

        self.action_dim = action_dim
        self.env = env

        # Buffer
        self.buffer_indexer = buffer_indexer

        self.state_buffer = np.zeros((buffer_indexer.size, *state_dim), dtype=np.float32)
        self.next_states_buffer = np.zeros((buffer_indexer.size, *state_dim), dtype=np.float32)
        self.actions_buffer = np.zeros(buffer_indexer.size, dtype=np.int32)
        self.terminals_buffer = np.zeros(buffer_indexer.size, dtype=np.int32)
        self.rewards_buffer = np.zeros(buffer_indexer.size, dtype=np.float32)

        # Constants
        self.eps_strategy = eps_strategy
        self.gamma = gamma
        self.update_model_frequency = update_model_frequency
        self.lr = lr

        # Network
        self.loss_function = nn.MSELoss()
        self.device = device

        # Default values
        self.optimizer = None
        self.q_model, self.target_model = None, None
        self.q_action_selector, self.target_action_selector = None, None

        model = self.__get_model__()
        self.set_q_model(model=model)
        self.set_target_model(model=deepcopy(model))

    def choose_eps_action(self, state):
        """
        Represents eps-greedy selection according to given eps-strategy.
        """
        if self.eps_strategy.check_random_prob():
            return np.random.randint(0, self.action_dim)
        else:
            action, _ = self.choose_action(state)
            return action

    def choose_action(self, state):
        """
        :return: (best action, Q-value for best action)
        """
        return self.q_action_selector.choose_action(state)

    def store_transition(self, state, next_state, action, reward, done):
        """
        Stores environment transition in buffer.
        """
        index = self.buffer_indexer.index()

        self.state_buffer[index] = state
        self.next_states_buffer[index] = next_state
        self.actions_buffer[index] = action
        self.rewards_buffer[index] = reward
        self.terminals_buffer[index] = done

        self.buffer_indexer.increment_index()

    def learn(self, episode):
        """
        Trains Q-function net.
        Using fixed target-model to predict target Q-function and q-model to choose actions while training.
        """
        if not self.buffer_indexer.is_batch_ready:
            return

        self.optimizer.zero_grad()

        states, next_states, actions, rewards, terminals = self.__sample_batch__()

        q_eval = self.q_model(states)[self.buffer_indexer.batch_indices, actions]

        with torch.no_grad():
            q_future = self.target_model(next_states)
            q_target = rewards + self.gamma * torch.max(q_future, dim=1).values * terminals

        self.__fit_network__(q_eval=q_eval, q_target=q_target)

        if episode % self.update_model_frequency == 0:
            self.set_target_model(model=deepcopy(self.q_model))

    def set_q_model(self, model):
        self.q_model = model.to(self.device)
        self.q_action_selector = ActionSelector(model=model, device=self.device)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.lr)

    def set_target_model(self, model):
        self.target_model = model.to(self.device)
        self.target_action_selector = ActionSelector(model=self.q_model, device=self.device)

    def __get_model__(self):
        neurons = 128
        return nn.Sequential(
            nn.Linear(self.state_dim, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, self.action_dim)
        )

    def __sample_batch__(self):
        """
        :return: ( [states], [next_states], [actions], [rewards], [is_done] )
        """
        indices = self.buffer_indexer.sample_indices()
        return [
            torch.tensor(self.state_buffer[indices]).to(self.device).float(),
            torch.tensor(self.next_states_buffer[indices]).to(self.device).float(),
            torch.tensor(self.actions_buffer[indices]).to(self.device).long(),
            torch.tensor(self.rewards_buffer[indices]).to(self.device).float(),
            torch.tensor(self.terminals_buffer[indices]).to(self.device)
        ]

    def __fit_network__(self, q_eval, q_target):
        error = self.loss_function(q_eval, q_target).to(self.device)
        error.backward()
        torch.nn.utils.clip_grad_value_(self.q_model.parameters(), clip_value=1)
        self.optimizer.step()


class AgentTrainer:
    def __init__(self,
                 agent: Agent,
                 reward_function=velocity_potentials,
                 episodes=400,
                 verbose=True):
        self.agent = agent
        self.episodes = episodes
        self.verbose = verbose
        self.reward_function = reward_function

    def train(self, path='trained_models/car_model.pt', render_step=None):
        rewards = np.zeros(self.episodes)
        for episode in np.arange(self.episodes):
            state = self.agent.env.reset()
            done = False
            episode_reward, max_position = 0, -1

            while not done:
                action = self.agent.choose_eps_action(state)
                next_state, reward, done, _ = self.agent.env.step(action)
                episode_reward += reward

                if render_step is not None and episode % render_step == 0:
                    self.agent.env.render()

                self.agent.store_transition(state=state,
                                            next_state=next_state,
                                            action=action,
                                            done=done,
                                            reward=self.reward_function(state, next_state, reward))

                max_position = max(max_position, next_state[0])
                self.agent.learn(episode=episode)
                state = next_state

            rewards[episode] = episode_reward

            self.agent.eps_strategy.decrease()
            if self.verbose:
                print("game {:03d} | reward {}{:04f} | eps {:05f} | max-position {:04f}".format(
                    episode + 1,
                    '-' if episode_reward < 0 else '+',
                    abs(episode_reward),
                    self.agent.eps_strategy.eps,
                    max_position)
                )

        self.agent.set_target_model(model=deepcopy(self.agent.q_model))
        torch.save(self.agent.target_model, path)
        return rewards


class AgentEnsemble:
    def __init__(self,
                 n_agents,
                 n_games,
                 state_dim,
                 action_dim,
                 reward_function=velocity_potentials,
                 verbose=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.n_agents = n_agents
        self.n_games = n_games
        self.device = device
        self.verbose = verbose
        self.trainers = [
            AgentTrainer(agent=Agent(state_dim=state_dim,
                                     action_dim=action_dim,
                                     device=device),
                         episodes=n_games,
                         reward_function=reward_function,
                         verbose=verbose)
            for _ in range(n_agents)
        ]

    def train(self, path='trained_models/ensembles/car_agent'):
        rewards = np.zeros((self.n_agents, self.n_games))
        for i, t in enumerate(self.trainers):
            if self.verbose:
                print('\n~~~~~~ Agent {} ~~~~~~\n'.format(i + 1))

            rewards[i] = t.train(path='{}{}.pt'.format(path, i))

        return rewards.mean(axis=0)

    def choose_action(self, state):
        counter = defaultdict(lambda: (0, 0))
        actions_ensemble = [t.agent.choose_action(state) for t in self.trainers]

        for (action, q_value) in actions_ensemble:
            amount, q = counter[action]
            counter[action] = (amount + 1, q + q_value)

        best_action = max(counter, key=lambda k: counter[k])
        amount, q_max = counter.get(best_action)
        return best_action, q_max / amount


if __name__ == '__main__':
    agents_ensemble = AgentEnsemble(n_agents=3, n_games=150, state_dim=[2], action_dim=3,
                                    reward_function=velocity_scaler,
                                    verbose=True)
    rs = agents_ensemble.train()
    #
    # plt.plot(range(len(rs)), rs)
    # plt.xlabel('game')
    # plt.ylabel('mean reward')
    # plt.show()

    # agent = Agent(state_dim=[2], action_dim=3)
    # trainer = AgentTrainer(agent=agent, episodes=200, reward_function=velocity_potentials)
    #
    # rs = trainer.train()

    plt.figure(figsize=(16, 9))
    plt.grid()
    plt.plot(range(len(rs)), rs, color='purple', linestyle='--', linewidth=1.5)
    plt.xlabel('game')
    plt.ylabel('mean reward')
    plt.savefig('plots/1-car-agent')
    plt.show()
    #
    # m = load_selector('trained_models/car_model.pt')
    # #
    # n = 1000
    # rs, values = interact(model=m, games=n)

    # plt.plot(range(n_games), rewards, color='green')
    # plt.plot(range(n_games), values, color='red')
    # plt.show()
