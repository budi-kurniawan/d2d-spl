import numpy as np
import gym
import pickle

FOURTHIRDS = 1.3333333333333
ONE_DEGREE = 0.0174532 # 2pi/360
SIX_DEGREES = 0.1047192
TWELVE_DEGREES = 0.2094384
FIFTY_DEGREES = 0.87266

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def select_action(theta, state, actions):
    discrete_state = discretise_state(state);
    prob = softmax(theta[discrete_state])
    return np.random.choice(actions, p=prob)

def load_policy(path):
    file = open(path, 'rb')
    theta = pickle.load(file)
    w = pickle.load(file)
    return theta, w

""" Reimplementation of http://incompleteideas.net/sutton/book/code/pole.c """
def discretise_state(state):
    x, x_dot, theta, theta_dot = state
    discrete_x = 0 if x < -0.8 else (1 if x < 0.8 else 2)
    discrete_x_dot = 0 if x_dot < -0.5 else (1 if x_dot < 0.5 else 2)
    discrete_theta_dot = 0 if theta_dot < -FIFTY_DEGREES else (1 if theta_dot < FIFTY_DEGREES else 2)
    
    discrete_theta = None
    if theta < -SIX_DEGREES:
        discrete_theta = 0
    elif theta < -ONE_DEGREE:
        discrete_theta = 1
    elif theta < 0:
        discrete_theta = 2
    elif theta < ONE_DEGREE:
        discrete_theta = 3
    elif theta < SIX_DEGREES:
        discrete_theta = 4
    else:
        discrete_theta = 5;
    return (discrete_theta * 3 * 3 * 3 + discrete_theta_dot * 3 * 3 + discrete_x_dot * 3 + discrete_x)

def create_cartpole_env():
    id = 'CartPole-v2'
    gym.envs.register(
        id=id,
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100_000
    )
    env = gym.make(id)
    return env
    
    