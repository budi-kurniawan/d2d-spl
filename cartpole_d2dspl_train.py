import sys
import os
import random
import time
from datetime import datetime
import numpy as np
import pickle
import gym
from cartpole_utils import discretise_state, select_action, create_cartpole_env
from cartpole_classifier import create_classifier

NUM_EPISODES = 1000
NUM_SAMPLES_FOR_CLASSIFIER = 50

NUM_STATES = 162
NUM_ACTIONS = 2
NUM_INPUTS = 4 # x, x_dot, theta, theta_dot
ALPHA_THETA = 0.5
ALPHA_W = 0.5
GAMMA = 0.95
LAMBDA_THETA = 0.9
LAMBDA_W = 0.8

def decay_traces(z_theta, z_w):
    z_theta *= LAMBDA_THETA
    z_w *= LAMBDA_W

def update_weights(rhat, theta, w, z_theta, z_w):
    w += ALPHA_W * rhat * z_w
    theta += ALPHA_THETA * rhat * z_theta

def save_policy(path, theta, w):
    file = open(path,'wb')
    pickle.dump(theta, file)
    pickle.dump(w, file)

def save_stats(path, state_stats, state_visits, theta):
    file = open(path,'w')
    for i in range(NUM_STATES):
        if state_visits[i] != 0:
            state_stats[i] /= state_visits[i]
            s1 = np.array2string(state_stats[i], separator=',', precision=4)
            s2 = np.array2string(theta[i], separator=',', precision=4)
            file.write(str(i) + ',' + s1 + ',' + s2 + '\n')
    file.close()

def load_policy(path):
    file = open(path, 'rb')
    theta = pickle.load(file)
    print(len(theta))
    w = pickle.load(file)
    
def run_episode(episode_no, state, actions, env, theta, w, z_theta, z_w, state_stats, state_visits):
    ep_reward = 0
    while True:
        action = select_action(theta, state, actions)
        next_state, reward, terminal, _ = env.step(action)
        discrete_state = discretise_state(state)
        
        if state_stats is not None:
            state_visits[discrete_state] += 1
            state_stats[discrete_state] += state

        old_prediction = w[discrete_state]
        prediction = 0.0 if terminal else w[discretise_state(next_state)]
        delta = reward + GAMMA * prediction - old_prediction
        if not terminal:
            z_theta[discrete_state][action] += 0.05 # strengthen the trace for the current state
            z_w[discrete_state] += 0.2 # strengthen the trace for the current state
            decay_traces(z_theta, z_w)
        update_weights(delta, theta, w, z_theta, z_w)
        state = next_state
        ep_reward += reward
        if terminal:
            break
    return ep_reward
        
def run_trial(trial_no, env, results_path):
    file = open(results_path + '/all-scores-' + str(trial_no).zfill(2) + '.txt','w')
    
    theta = np.zeros([NUM_STATES, NUM_ACTIONS], dtype=np.float64)
    w = np.zeros(NUM_STATES, dtype=np.float64)
    z_theta = np.zeros([NUM_STATES, NUM_ACTIONS], dtype=np.float64)
    z_w = np.zeros(NUM_STATES, dtype=np.float64)
    actions = np.arange(NUM_ACTIONS) # don't convert to torch

    total = 0
    start_time = datetime.now()
    buffer = []
    
    for i_episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        # reset traces
        z_theta[:] = 0.0; z_w[:] = 0.0
        state_stats = np.zeros([NUM_STATES, NUM_INPUTS], dtype=np.float64)
        state_visits = np.zeros(NUM_STATES, dtype=np.int32)
        ep_reward = run_episode(i_episode, state, actions, env, theta, w, z_theta, z_w, state_stats, state_visits)
        total += ep_reward
        
        buffer.append((i_episode, ep_reward, state_stats, state_visits))

        file.write(str(i_episode) + "," + str(ep_reward) + '\n')
        
    buffer.sort(key=lambda tup: tup[1], reverse=True) # sorted by reward, biggest on top
    del buffer[NUM_SAMPLES_FOR_CLASSIFIER : ] # only use the first n samples
    consolidated_state_stats = np.zeros([NUM_STATES, NUM_INPUTS], dtype=np.float64)
    consolidated_state_visits = np.zeros(NUM_STATES, dtype=np.int32)
    for i in range(len(buffer)):
        ep, r, state_stats, state_visits = buffer[i]
        consolidated_state_stats += state_stats
        consolidated_state_visits += state_visits
        
    end_time = datetime.now()
    delta = end_time - start_time
    print('first ' + str(NUM_EPISODES) + ' episodes learning took ' + str(delta.total_seconds()) + ' seconds')

    print('trial', trial_no, ', avg score:', total / NUM_EPISODES)
    
    save_policy(results_path + '/policy' + str(trial_no).zfill(2) + '-' + str(NUM_EPISODES) + '.p', theta, w)
    save_stats(results_path + '/trainingset' + str(trial_no).zfill(2) + '.txt', consolidated_state_stats, consolidated_state_visits, theta)
    
    # for 2nd batch of episodes
    start_time = datetime.now()
    for i_episode in range(NUM_EPISODES + 1, NUM_EPISODES * 2 + 1):
        state = env.reset()
        z_theta[:] = 0.0; z_w[:] = 0.0
        ep_reward = run_episode(i_episode, state, actions, env, theta, w, z_theta, z_w, None, None)
        total += ep_reward
        # update cumulative reward
        file.write(str(i_episode) + "," + str(ep_reward) + '\n')
    end_time = datetime.now()
    delta = end_time - start_time
    print('second ' + str(NUM_EPISODES) + ' episodes learning took ' + str(delta.total_seconds()) + ' seconds')

    print('trial', trial_no, ', avg score:', total / NUM_EPISODES)
    file.close()
    save_policy(results_path + '/policy' + str(trial_no).zfill(2) + '-' + str(NUM_EPISODES * 2) + '.p', theta, w)
    
if __name__ == '__main__':
    env = create_cartpole_env()
    results_path = 'results/d2d_results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    num_trials = 10
    print('start:', datetime.now().strftime('%d/%m/%y %H:%M:%S'))
    for trial in range(num_trials):
        print('\ntrial', trial)
        env.seed(trial)
        np.random.seed(trial)
        policy_path = results_path + '/policy' + str(trial).zfill(2) + '-' + str(NUM_EPISODES) + '.p'
        run_trial(trial, env, results_path)
        start_time = datetime.now()
        create_classifier(trial)
        end_time = datetime.now()
        delta = end_time - start_time
        print('Classifier learning took ' + str(delta.total_seconds()) + ' seconds')
    print('end:', datetime.now().strftime('%d/%m/%y %H:%M:%S'))
    