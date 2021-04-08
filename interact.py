import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.action_selector import load_selector


def interact(model, games=20, env=gym.make('MountainCar-v0')):
    game_rewards = np.zeros(games)
    for i in tqdm(np.arange(games)):
        state = env.reset()
        done = False
        game_reward = 0
        while not done:
            action, q_value = model.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            game_reward += reward
            state = next_state
            env.render()

        game_rewards[i] = game_reward

    return game_rewards


if __name__ == '__main__':
    m = load_selector('trained_models/car_model.pt')

    n_games = 20
    rewards = interact(model=m, games=n_games)

    plt.figure(figsize=(16, 9))
    plt.grid()
    plt.plot(range(n_games), rewards, color='green')
    plt.xlabel('game')
    plt.ylabel('reward')
    plt.savefig('plots/car-agent-interaction')
    plt.show()
