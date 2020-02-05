import sys
import os
import random
import time
from pathlib import Path
from datetime import datetime
import torch
import torch.nn

import numpy as np
#from itertools import count
#from collections import namedtuple
import pickle
import torch.multiprocessing as mp
import gym
#from cartpole_utils import create_cartpole_env
from cartpole_a3c_train import Net, v_wrap

num_test_episodes = 100
input_dim = 4

def _to_variable(x: np.ndarray) -> torch.Tensor:
    return torch.autograd.Variable(torch.Tensor(x))

def get_Q(model, states: np.ndarray) -> torch.FloatTensor:
    states = _to_variable(states.reshape(-1, input_dim))
    model.train(mode=False)
    return model(states)

def select_test_action(model, states):
    model.train(mode=False)
    scores = get_Q(model, states)
    _, argmax = torch.max(scores.data, 1)
    return int(argmax.numpy())

def test(trial, model, env):
    total = 0
    success = 0
    num_inputs = env.observation_space.shape[0]
    for ep in range(1, num_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        while True:
            action = model.select_action(v_wrap(state[None, :]))
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        #print('episode', ep, ', total reward: ', ep_reward)
        total += ep_reward
        if ep_reward == 100_000:
            success += 1
    print('=== avg score for trial ' + str(trial) + ': ' + str(total / num_test_episodes) + ", successes:" + str(success))

if __name__ == '__main__':
    num_workers = min(4, mp.cpu_count())
    num_episodes = 2000 #num_workers * 1_000
    env = gym.make('CartPole-v2').unwrapped #create_cartpole_env()
    num_trials = 10
    for trial in range(num_trials):
        env.seed(trial)
        model_path = 'a3c_results/model' + str(trial).zfill(2) + '-' + str(num_episodes) + '.p'
        model = pickle.load(open(model_path, "rb"))
        #model.exploration_rate = 0.0
        test(trial, model, env)