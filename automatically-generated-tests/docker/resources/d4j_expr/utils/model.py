import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dim, cuda=True, output_dim=1, lr=0.001):
        super(Model, self).__init__()

        device = torch.device("cuda:0" if cuda else "cpu")
        self.learn_step_counter = 0
        self.encoder = Encoder(input_dim=state_dim, hidden_dim=embed_dim).to(device)
        self.eval_net = Critic(input_dim=action_dim+embed_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
        self.target_net = Critic(input_dim=action_dim+embed_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
        self.optimizer = torch.optim.Adam([     # only optimize eval_net and embedding 
                                            {'params':self.eval_net.parameters(), 'lr': lr},
                                            {'params':self.encoder.parameters(), 'lr': lr},
                                            ])

class Critic(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2, bias=True)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_dim*2, output_dim, bias=True)
        self.fc3.weight.data.normal_(0, 0.1)
        # self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        # self.fc2.weight.data.normal_(0, 0.1)
    
    def forward(self, input):
        out1 = F.relu(self.fc1(input))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)

        # out2 = torch.tanh(self.fc2(out1))
        return out3



class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc2.weight.data.normal_(0, 0.1)
    
    def forward(self, input):
        out1 = F.relu(self.fc1(input))
        out2 = F.relu(self.fc2(out1))
        # out2 = F.relu(self.fc2(out1))
        return out2