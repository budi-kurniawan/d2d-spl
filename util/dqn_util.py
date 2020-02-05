# An implementation of the original DQN (with no target network)

import random
from random import randrange
from collections import namedtuple
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

min_epsilon = 0.01
batch_size = 64
gamma = 0.99

class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(DQN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayMemory(object):
    def __init__(self, size: int) -> None:
        self.size = size
        self.cursor = 0
        self.memory = []

    def push(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        if len(self) < self.size:
            self.memory.append(Transition(state, action, reward, next_state, done))
        else:
            self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.size

    def pop(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent(object):
    def __init__(self, memory, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        self.memory = memory
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def select_action(self, states: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        states = torch.Tensor(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)
    
    def add_sample(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)
        
    def train(self) -> None:
        if len(self.memory) <= batch_size:
            return
        minibatch = self.memory.pop(batch_size)
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        Q_predict = self.get_Q(states)
        Q_target = Q_predict.clone().data.numpy() # Q_target is not a second network, most of its values are the same as the reward at the current timestep
        Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(self.get_Q(next_states).data.numpy(), axis=1) * ~done
        Q_target = torch.Tensor(Q_target)
        return self._train(Q_predict, Q_target)

    def _train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes loss and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss
    
    def before_episode(self):
        pass # do nothing here
        
    def after_episode(self):
        pass # do nothing here

class DQNWithTargetNetworkAgent(DQNAgent):
    def __init__(self, memory, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__(memory, input_dim, output_dim, hidden_dim)
        self.dqn2 = DQN(input_dim, output_dim, hidden_dim) # target network
    
    def before_episode(self):
        self.dqn2.load_state_dict(self.dqn.state_dict()) # copy weights from dqn1 to dqn2

    def get_Q2(self, states: np.ndarray):
        states = torch.Tensor(states.reshape(-1, self.input_dim))
        self.dqn2.train(mode=False)
        return self.dqn2(states)
    
    def train(self) -> None:
        if len(self.memory) <= batch_size:
            return
        minibatch = self.memory.pop(batch_size)
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        Q_predict = self.get_Q(states)
        Q_target = Q_predict.clone().data.numpy() # Q_target is not a second network, most of its values are the same as the reward at the current timestep
        Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(self.get_Q2(next_states).data.numpy(), axis=1) * ~done
        Q_target = torch.Tensor(Q_target)
        return self._train(Q_predict, Q_target)

class DoubleDQNAgent(DQNAgent):
    def __init__(self, memory, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__(memory, input_dim, output_dim, hidden_dim)
        self.dqn1 = self.dqn
        self.dqn2 = DQN(input_dim, output_dim, hidden_dim)
    
    def before_episode(self):
        self.dqn2.load_state_dict(self.dqn1.state_dict()) # copy weights from dqn1 to dqn2

    def random_0_or_1(self):
        return randrange(2)

    def select_action(self, states: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            r = self.random_0_or_1()
            self.dqn = self.dqn1 if r == 0 else self.dqn2
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_Q1(self, states: np.ndarray):
        self.dqn = self.dqn1
        return self.get_Q(states)

    def get_Q2(self, states: np.ndarray):
        self.dqn = self.dqn2
        return self.get_Q(states)
    
    def train(self) -> None:
        if len(self.memory) <= batch_size:
            return
        minibatch = self.memory.pop(batch_size)
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        r = self.random_0_or_1()
        Q_predict = self.get_Q1(states) if r == 0 else self.get_Q2(states)
        Q_target = Q_predict.clone().data.numpy() # Q_target is not a second network, most of its values are the same as the reward at the current timestep
        if r == 0:
            Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(self.get_Q2(next_states).data.numpy(), axis=1) * ~done
        else:
            Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(self.get_Q1(next_states).data.numpy(), axis=1) * ~done
        Q_target = torch.Tensor(Q_target)
        return self._train(Q_predict, Q_target)

class PERAgent(DoubleDQNAgent):
    
    
    #def __init__(self, memory, input_dim: int, output_dim: int, hidden_dim: int) -> None:
    #    super().__init__(memory, input_dim, output_dim, hidden_dim)
    def __init__(self, memory, input_dim: int, output_dim: int, hidden_dim):
        super().__init__(memory, input_dim, output_dim, hidden_dim)
        self.state_size = input_dim
        self.action_size = output_dim

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 20000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 64
        self.train_start = 1000

    def add_sample(self, error: float, transition) -> None:
        self.memory.push(error, transition)

    def train(self):
        if self.memory.tree.n_entries < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        #mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch, idxs, is_weights = self.memory.pop(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        #states = Variable(states).float()
        #pred = self.model(states)
        pred = self.dqn1(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        errors = torch.abs(pred - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()
        self.memory.step()
