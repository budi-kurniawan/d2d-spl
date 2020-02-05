import sys
import os
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
#from itertools import count
from collections import namedtuple
from sklearn.neural_network import MLPClassifier
from datetime import datetime
import time
import pickle
import gym
from cartpole_utils import create_cartpole_env, discretise_state, load_policy

num_episodes = 100

def select_action_mlp(classifier, state):
    p = classifier.predict([state])
    return p[0]

def test_mlp(trial, classifier, env):
    total = 0
    success = 0
    for ep in range(1, num_episodes + 1):
        ep_reward = 0
        state = env.reset()
        while True:
            action = select_action_mlp(classifier, state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        #print('episode', ep, ', total reward: ', ep_reward)
        total += ep_reward
        if ep_reward == 100_000:
            success += 1
    print('=== avg score for trial ' + str(trial) + ': ' + str(total / num_episodes) + ", successes:" + str(success))

def test_policy(trial, theta, env):
    total = 0
    success = 0
    for ep in range(1, num_episodes + 1):
        ep_reward = 0
        state = env.reset()
        while True:
            discrete_state = discretise_state(state)
            action = np.argmax(theta[discrete_state])
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        #print('episode', ep, ', total reward: ', ep_reward)
        total += ep_reward
        if ep_reward == 100_000:
            success += 1
    print('=== avg score for trial ' + str(trial) + ': ' + str(total / num_episodes) + ", successes:" + str(success))

if __name__ == '__main__':
    env = create_cartpole_env()
    num_trials = 10
    print('test mlp')
    for trial in range(num_trials):
        env.seed(trial)
        model_path = 'results/cartpole-classifier' + str(trial).zfill(2) + '.p'
        classifier = pickle.load(open(model_path, "rb"))
        test_mlp(trial, classifier, env)

    quit()
    print('test policy 1000')
    for trial in range(num_trials):
        env.seed(trial)
        policy_path = 'results/policy' + str(trial).zfill(2) + '-1000.p'
        theta, w = load_policy(policy_path)
        test_policy(trial, theta, env)
        
    print('\ntest policy 2000')
    for trial in range(num_trials):
        env.seed(trial)
        policy_path = 'results/policy' + str(trial).zfill(2) + '-2000.p'
        theta, w = load_policy(policy_path)
        test_policy(trial, theta, env)
        
