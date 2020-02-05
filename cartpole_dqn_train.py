import os
import torch
import torch.nn
import numpy as np
import random
import gym
import pickle
from collections import namedtuple
from datetime import datetime
from typing import List, Tuple
from util.dqn_util import DQNAgent, ReplayMemory

NUM_EPISODES = 1000
min_epsilon = 0.01
batch_size = 64
gamma = 0.99
hidden_dim = 12
mem_size = 50_000

def play_episode(env: gym.Env, agent, eps: float) -> int:
    agent.before_episode()
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        a = agent.select_action(s, eps)
        s2, r, done, _ = env.step(a)
        if done:
            r = -1
        total_reward += r
        agent.add_sample(s, a, r, s2, done)
        agent.train()
        s = s2
    agent.after_episode()
    return total_reward

def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    return input_dim, output_dim

def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

def save_model(path, dqn):
    file = open(path,'wb')
    pickle.dump(dqn, file)

def main(env, input_dim, output_dim, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    print('start:\n\b', datetime.now().strftime('%d/%m/%y %H:%M:%S'))
    print('Trial,Learning Time 1 (seconds),Learning Time 2 (seconds),Avg Score 1,Avg Score 2')
    print('=================================================================================')
    num_trials = 10
    for trial in range(num_trials):
        env.seed(trial)
        random.seed(trial)
        torch.manual_seed(trial)
        np.random.seed(trial)
        file = open(results_path + '/all-scores-' + str(trial).zfill(2) + '.txt','w')
        start_time = datetime.now()

        memory = ReplayMemory(mem_size)
        agent = DQNAgent(memory, input_dim, output_dim, hidden_dim)
        total = 0        
        for episode in range(1, NUM_EPISODES + 1):
            eps = epsilon_annealing(episode, NUM_EPISODES - 1, min_epsilon)
            ep_reward = play_episode(env, agent, eps)
            total += ep_reward
            print("episode:", episode, ', Reward:', ep_reward)
            file.write(str(episode) + "," + str(ep_reward) + '\n')
        
        end_time = datetime.now()
        delta1 = end_time - start_time
        save_model(results_path + '/model' + str(trial).zfill(2) + '-' + str(NUM_EPISODES) + '.p', agent.dqn)

        ### 2nd batch
        start_time = datetime.now()
        for episode in range(1 + NUM_EPISODES, 2 * NUM_EPISODES + 1):
            eps = epsilon_annealing(episode, NUM_EPISODES - 1, min_epsilon)
            ep_reward = play_episode(env, agent, eps)
            total += ep_reward
            #print("episode:", episode, ', Reward:', ep_reward)
            file.write(str(episode) + "," + str(ep_reward) + '\n')
        file.close()
        
        end_time = datetime.now()
        delta2 = end_time - start_time
        print(str(trial) + ',' + str(delta1.total_seconds()) + ',' + str(delta2.total_seconds()) + ',' 
              + str(total / NUM_EPISODES) + ',' + str(total / (2*NUM_EPISODES)))
        save_model(results_path + '/model' + str(trial).zfill(2) + '-' + str(2*NUM_EPISODES) + '.p', agent.dqn)

if __name__ == '__main__':
    from cartpole_utils import create_cartpole_env
    env = create_cartpole_env()
    input_dim, output_dim = get_env_dim(env)
    results_path = 'results/dqn_results'
    main(env, input_dim, output_dim, results_path)