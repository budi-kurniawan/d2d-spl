"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
https://github.com/MorvanZhou/pytorch-A3C
"""
import os
from datetime import datetime
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
num_workers = min(4, mp.cpu_count())
NUM_EPISODES = 2_000

gym.envs.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=100_000
)

num_state_variables = 4
num_actions = 2
#hidden_layer_sizes = [200, 100]
hidden_layer_sizes = [100, 50]

class Net(nn.Module):
    def __init__(self, num_state_variables, num_actions):
        super(Net, self).__init__()
        self.pi1 = nn.Linear(num_state_variables, hidden_layer_sizes[0])
        self.pi2 = nn.Linear(hidden_layer_sizes[0], num_actions)
        self.v1 = nn.Linear(num_state_variables, hidden_layer_sizes[1])
        self.v2 = nn.Linear(hidden_layer_sizes[1], 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def select_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, trial, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(num_state_variables, num_actions) # local network
        self.env = gym.make('CartPole-v2').unwrapped
        self.env.seed(trial)

    def run(self):
        total_step = 1
        while self.g_ep.value <= NUM_EPISODES:
            print('start episode ', self.g_ep.value)
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                a = self.lnet.select_action(v_wrap(state[None, :]))
                next_state, r, done, _ = self.env.step(a)
                #if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(state)
                buffer_r.append(r)

                if done or total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, next_state, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                state = next_state
                total_step += 1
        self.res_queue.put(None)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    #res_queue.put(global_ep_r.value)
    res_queue.put(ep_r)
    #print(name, "Ep:", global_ep.value, "| Ep_r: %.4f" % global_ep_r.value,)
    print(name, "Ep:", global_ep.value, "| Ep_r: %.4f" % ep_r)

def save_model(path, model):
    file = open(path, 'wb')
    pickle.dump(model, file)
    file.close()

def run_trial(trial, results_path):
    gnet = Net(num_state_variables, num_actions) # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # parallel training
    start_time = datetime.now()
    
    workers = [Worker(trial, gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(num_workers)]
    [w.start() for w in workers]
    res = [] # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    end_time = datetime.now()
    delta = end_time - start_time
    print('Trial', 0, ', A3C learning 2 took ' + str(delta.total_seconds()) + ' seconds')
    
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()
    score_file = open(results_path + '/all-scores-' + str(trial).zfill(2) + '.txt','w')
    for i in range(1, len(res)):
        score_file.write(str(i) + ',' + str(res[i - 1]) + '\n')
    score_file.close()
    model_path = results_path + '/model' + str(trial).zfill(2) + '-' + str(NUM_EPISODES) + '.p'
    save_model(model_path, gnet)
                        
if __name__ == "__main__":
    results_path = 'results/a3c_results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    num_trials = 10
    times_file = open(results_path + '/times.txt', 'a+')
    for trial in range(num_trials):
        print('\ntrial', trial)
        np.random.seed(trial)
        start_time = datetime.now()
        run_trial(trial, results_path)
        end_time = datetime.now()
        delta = end_time - start_time
        times_file.write('=== trial ' + str(trial) + '\n')
        times_file.write('Learning took ' + str(delta.total_seconds()) + ' seconds\n\n')
    print('end:', datetime.now().strftime('%d/%m/%y %H:%M:%S'))
    times_file.close()