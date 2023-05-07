import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np


def getState(X, methods):
    """
    input: the cov matrix of current tests
    output: the state feature of the current matrix
    """
    if torch.cuda.is_available():
        X = X.cuda()

    target = X[0,:] > 0    # only consider those covered by failing tests, to speed up (originally do not have)
    target_index = torch.where(X[0,:] > 0)[0].cpu().numpy().tolist()
    X = X[:, target]
    method_class = []
    for index in target_index:
        method_class.append(methods[index])

    unique, indices, counts = torch.unique(X, sorted=False, return_inverse=True, return_counts=True, dim=1)
    num_ag = unique.shape[1]   # number of ambiguity group

    impurity = 1  # calculate the gini index of the most suspicious ambiguity group
    group_cover_min = 10
    for j in range(unique.shape[1]):  # calculate impurity for each group
        group_cover_test_num = unique[:, j].sum()    # number of covered test for this group
        if group_cover_test_num > group_cover_min:
            continue

        group_index = torch.where(indices == j)[0].cpu().numpy().tolist()
        group_size = counts[j]
        group_cover_min = group_cover_test_num
        group_method_class = []
        for index in group_index:
            group_method_class.append(method_class[index])
        dic = dict(Counter(group_method_class))
        impurity = 1
        for value in dic.values(): 
            impurity -= (value / group_size) * (value / group_size)
        
    uncovered = X.shape[1] - torch.sum(X.sum(0)>0)
    num_tests = X.shape[0] - 1
    # return torch.tensor([num_tests, num_ag, impurity, uncovered]).float()
    return torch.tensor([num_tests, num_ag, uncovered]).float()

def getFeature(X, x):
    if x is None:
        return .0

    # calculate cover, i.e., the jaccard coeff with initial failing test
    cover = 1 - np.mean(cdist(X[[0], :], torch.unsqueeze(x, 0), 'jaccard')) 
    _, n = X.shape

    # calculate spilt
    # target = kwargs['target']
    target = X[0,:] > 0    # only consider those covered by failing tests, to speed up (originally do not have)
    X = X[:, target]
    x = x[target]

    if torch.cuda.is_available():
        X = X.cuda()
        x = x.cuda()

    unique, indices, counts = torch.unique(X, sorted=False, return_inverse=True, return_counts=True, dim=1)
    split = 0
    for j in range(unique.shape[1]):  # calculate values for each ambiguity group
        group_index = (indices == j)
        group_prio = torch.sum(group_index) / n  # |g|/n
        div = min(int(torch.sum(x[group_index] == 0)), int(torch.sum(x[group_index] == 1))) 
        split += div * group_prio
    return torch.tensor([split, cover]).float()

def getallFeature(X_cur, X_all):
    # if x is None:
    #     return .0

    # calculate cover, i.e., the jaccard coeff with initial failing test
    cover = torch.tensor(1 - (cdist(X_cur[[0], :], X_all, 'jaccard'))).transpose(0,1).cuda() # [m,1]
    m, n = X_all.shape

    # calculate spilt
    # target = kwargs['target']
    target = X_cur[0,:] > 0    # only consider those covered by failing tests, to speed up (originally do not have)
    X_cur = X_cur[:, target]
    X_all = X_all[:, target]

    if torch.cuda.is_available():
        X_cur = X_cur.cuda()
        X_all = X_all.cuda()

    unique, indices, counts = torch.unique(X_cur, sorted=False, return_inverse=True, return_counts=True, dim=1)
    split = torch.zeros(m,1).cuda()
    for j in range(unique.shape[1]):  # calculate values for each ambiguity group
        group_index = (indices == j)
        group_prio = torch.sum(group_index) / n  # |g|/n
        stack = torch.stack((torch.sum(X_all[:,group_index]==0, dim=1), torch.sum(X_all[:,group_index]==1, dim=1)),0) 
        # print(stack.shape)
        div = torch.min(stack, 0)[0].unsqueeze(1)
        split += div * group_prio
    return torch.cat([split, cover], dim=1).float()

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