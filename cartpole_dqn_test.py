import sys
import os
import random
import time
from pathlib import Path
from datetime import datetime
import torch
import torch.nn

import numpy as np
import pickle
from util.dqn_util import DQN
from cartpole_utils import create_cartpole_env

num_test_episodes = 100
input_dim = 4

def _to_variable(x: np.ndarray) -> torch.Tensor:
    return torch.autograd.Variable(torch.Tensor(x))

def get_Q(dqn, states: np.ndarray) -> torch.FloatTensor:
    states = _to_variable(states.reshape(-1, input_dim))
    dqn.train(mode=False)
    return dqn(states)

def select_test_action(dqn, states):
    dqn.train(mode=False)
    scores = get_Q(dqn, states)
    _, argmax = torch.max(scores.data, 1)
    return int(argmax.numpy())

def test_dqn(trial, dqn, env):
    total = 0
    success = 0
    num_inputs = env.observation_space.shape[0]
    for ep in range(1, num_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        while True:
            state = np.reshape(state, [1, num_inputs])
            action = select_test_action(dqn, state)
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
    num_episodes = 1000
    model_parent_path = 'results/double_dqn_results_1'
    env = create_cartpole_env()
    num_trials = 10
    for trial in range(num_trials):
        env.seed(trial)
        model_path = model_parent_path + '/model' + str(trial).zfill(2) + '-1000.p'
        dqn = pickle.load(open(model_path, "rb"))
        dqn.exploration_rate = 0.0
        test_dqn(trial, dqn, env)
        
    for trial in range(num_trials):
        env.seed(trial)
        model_path = model_parent_path + '/model' + str(trial).zfill(2) + '-2000.p'
        dqn = pickle.load(open(model_path, "rb"))
        dqn.exploration_rate = 0.0
        test_dqn(trial, dqn, env)
        