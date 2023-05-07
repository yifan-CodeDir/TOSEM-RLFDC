from turtle import hideturtle
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import sys
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import json
import math
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.getcwd() + "/..")
from Env import D4Jenv

MEMORY_SIZE = 2000   # 2000
EPSILON = 0.9
TARGET_REPLACE_ITER = 100
BATCH_SIZE = 32
GAMMA = 0.9


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc1.weight.data.normal_(0, 0.1)
    
    def forward(self, state):
        out1 = F.relu(self.fc1(state))
        action_value = self.fc2(out1)
        return action_value

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1) 
    
    def forward(self, input):
        out, (h_n, c_n) = self.lstm(input)
        return h_n

class DQN(object):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, lr=0.01):

        # use cuda if cuda is available
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim, embed_dim).to(self.device)
        self.eval_net = Net(embed_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = Net(embed_dim, hidden_dim, output_dim).to(self.device)

        self.memory = [{} for _ in range(MEMORY_SIZE)]
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.state_dim = input_dim
        self.action_dim = output_dim
        self.memory_counter = 0
        self.learn_step_counter = 0

    def choose_action(self, state, mask):
        state = state.to(self.device)
        if np.random.uniform() < EPSILON:
            embeddings = self.encoder(state)                  # Get embedding by LSTM
            action_value = self.eval_net(embeddings)          # Get action
            action_value[:, mask] = -math.inf                    # mask tests that have been chosen
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action
    
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transition
        sample_index = np.random.choice(MEMORY_SIZE, BATCH_SIZE)
        
        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        b_m = []

        for i in range(len(sample_index)):       # collect memory
            index = sample_index[i]
            b_s.append(self.memory[index]['s'])  
            b_a.append(self.memory[index]['a'])
            b_r.append(self.memory[index]['r'])
            b_s_.append(self.memory[index]['s_'])
            b_m.append(self.memory[index]['m'])

        packed_s = pack_sequence(b_s, enforce_sorted=False).to(self.device)    # state has different length
        packed_s_ = pack_sequence(b_s_, enforce_sorted=False).to(self.device)

        embeddings_s = self.encoder(packed_s).squeeze(0)
        embeddings_s_ = self.encoder(packed_s_).squeeze(0)

        b_a = torch.tensor(b_a).unsqueeze(-1).to(self.device) # (batch_size, 1)
        b_r = torch.tensor(b_r).unsqueeze(-1).type(torch.FloatTensor).to(self.device) # (batch_size, 1)
        q_eval = self.eval_net(embeddings_s).gather(1, b_a).type(torch.FloatTensor).to(self.device) # (batch_size, n)
        q_next = self.target_net(embeddings_s_).detach().type(torch.FloatTensor).to(self.device)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1).type(torch.FloatTensor).to(self.device)
        
        # update params
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, s, a, r, s_, mask):
        # transition = np.hstack((s, [a, r], s_))
        transition = {'s':s, 'a':a, 'r':r, 's_':s_, 'm':mask}
        index = self.memory_counter % MEMORY_SIZE
        self.memory[index] = transition
        self.memory_counter += 1
    

# python DQN.py --project Chart --version 12 --load_path ../model/DQN_Chart-12.pt
# python DQN.py --project Lang --version 1 --train --episode 400
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, default=None)
    parser.add_argument("--version", '-n', type=int, default=None)
    parser.add_argument("--train", action='store_true', default=False)  
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--episode", type=int, default=None)

    """
        Defects4J Simulation
    """
    excluded = {
        'Lang': [2, 23, 56],
        'Chart': [],
        'Time': [21],
        'Math': [],
        'Closure': [63, 93]      
    }
    projects = {
        'Lang':    (1, 65),
        'Chart':   (1, 26),
        'Time':    (1, 27),
        'Math':    (1, 106),
        'Closure': (1, 133)
    }

    # get args
    args = parser.parse_args()
    p = args.project
    n = args.version
    train = args.train
    save_path = args.save_path if args.save_path is not None else f"../model/DQN_{p}-{n}.pt"
    load_path = args.load_path
    episode = args.episode

    # initalize environment
    env = D4Jenv(p, n, feat_dim=1024)
    num_failing_tests = env.num_failing_tests

    iter = 10          # select 10 additional tests in total

    # initialzie model
    if not train:     # if evaluation, then load pretrained model
        # dqn.load_state_dict(torch.load(f"../output/DQN_{p}-{n}.pt"))
        # dqn.eval()
        dqn = torch.load(load_path)
    else:
        dqn = DQN(input_dim=env.observation_space, embed_dim=512, hidden_dim=128, output_dim=env.action_space)


    log_reward = []
    log_rank = []

    for i_episode in range(episode):
        s = env.reset(0)         # select an initial failing test to reset environment
        ep_r = 0                 # record episode reward
        for _ in range(iter):

            a = dqn.choose_action(s, env.mask)
            # take action
            s_, r = env.step(a)

            dqn.store_transition(s, a, r, s_, env.mask)

            ep_r += r
            if dqn.memory_counter > MEMORY_SIZE:
                dqn.learn()
            
            s = s_
        
        print('Ep: ', i_episode,
            '| Ep_r: ', round(ep_r, 2),
            '| Min rank:', np.min(env.results["ranks"][-1]))

        # save results to output.json
        with open("output.json", "a") as f:
            json.dump(env.results, f, sort_keys=True,
                ensure_ascii=False, indent=4)
        print(f"* Results are saved to output.json")

        log_reward.append(ep_r)
        log_rank.append(np.min(env.results["ranks"][-1]))

    # save model 
    if train:
        torch.save(dqn, save_path)

    # plot figure
    plt.figure(figsize=(4, 4), dpi=300)

    x = range(episode)
    l_reward,  = plt.plot(x, log_reward)
    l_rank,  = plt.plot(x, log_rank)

    plt.legend(handles=[l_reward, l_rank], labels=['reward', 'rank'])
    plt.savefig(f'DQN_{p}-{n}.pdf')

            