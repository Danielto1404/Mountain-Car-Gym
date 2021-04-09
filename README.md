### Mountain Car Agent

Implementation of RL agent for **[gym mountain car environment](https://gym.openai.com/envs/MountainCar-v0)**

I've picked Deep-Q-Learning algorithm to train agent.

* Trained neural networks for q-function located in [trained models package](trained_models)
* Plots for learning graphics located in [plots package](plots)
* [interact.py](interact.py) - runner to visualize environment processes.
* [agents.py](agents.py) - Implementation of DQN-algorithm for RL.
* [Utils](core) - package of utils:
    * [buffer for store transitions](core/buffer_indexer.py)
        * Provides a simple interface to count amount of added elements and sample random batches with **fixed
          batch_size**
    * [epsilon-greedy strategy](core/eps_strategy.py)
        * Implements a manager for exploration-exploitation trade-off while learning.
            * start - epsilon from which learning starts
            * decay - linear decay of epsilon after each learning episode
            * min_eps - minimum value for epsilon
            * check_random_prob() - returns probability to force agent make a random action
    * [modified rewards functions](core/rewards.py)
        * Implementation of modified rewards for DQN agent.
            * naive - return reward (in our case -1 if not finished)
            * velocity_scaler - rewards an agent for big value of velocity
            * velocity_potentials - rewards an agent for maintaining velocity
    * [loading manager & best action selector](core/action_selector.py)
        * Implementation of **pytorch load** with ability to chose best action by given state

