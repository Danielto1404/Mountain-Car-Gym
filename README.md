### Mountain Car Agent

Agent implementation of **[gym mountain car enviroment](https://gym.openai.com/envs/MountainCar-v0)**

I've picked Deep-Q-Learning algorithm to train agent.

* Trained neural networks for q-function in [trained models package](trained_models)
* [interact.py](interact.py) - runner to visualize enviroment processes.
* [agents.py](agents.py) - Implementation of DQN-algorihhm for RL.
* [Utils](core) - package of implementation class like:
	* [buffer for store transitions](core/buffer_indexer.py)
	* [epsilon-greedy strategy](core/eps_strategy.py)
	* [modified rewards functions](core/rewards.py)
	* [loading manager & best action selector](core/action_selector.py)
