import torch
import torch.nn as nn
import sys

sys.path.append("..")
from utils.d4j import load
from utils.FL import *
from RL_algo.network import Critic, Encoder, Model, getFeature
from torch.nn.utils.rnn import pack_sequence
from sklearn.preprocessing import binarize
import tqdm

BATCH_SIZE = 64
GAMMA = 0.9

class Memory():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.pool = [{} for _ in range(memory_size)]       # initialize memory pool
        self.memory_counter = 0
    
    def store_transition(self, s, ds, a, r):
        transition = {'s':s, 'ds':ds, 'a':a, 'r':r}     # s=[current tests], a=feature of selected tests, r=relative promoted rank 
        index = self.memory_counter % self.memory_size
        self.pool[index] = transition
        self.memory_counter += 1

def learn(X, model, memory, batch_size):

    # random sample batch data from memory
    sample_index = np.random.choice(memory.memory_size, batch_size)
    b_s = []
    b_a = []
    b_r = []
    b_s_ = []

    # TODO: need to define state to describe the state of current matrix, besides, the dimension of state should be the same for each version
    for i in range(len(sample_index)):        # collect memory
        index = sample_index[i]
        s = torch.index_select(X, 0, memory.pool[index]['s'])
        ds = torch.index_select(X, 0, memory.pool[index]['ds'])
        a = memory.pool[index]['a']
        s_ = torch.cat([s,ds],0)

        b_s.append(s)   # (batch_size, ?)
        b_a.append(a)   # (batch_size, 2)
        b_r.append(memory.pool[index]['r'])   # (batch_size, 1)
        b_s_.append(s_)  # (batch_size, ?)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # use the memory to update parameters
    packed_s = pack_sequence(b_s, enforce_sorted=False).to(device)    # state has different length
    packed_s_ = pack_sequence(b_s_, enforce_sorted=False).to(device)

    embed_s = model.encoder(packed_s).squeeze(0)
    embed_s_ = model.encoder(packed_s_).squeeze(0)
    
    b_r = torch.tensor(b_r).unsqueeze(-1).type(torch.FloatTensor).to(device) # (batch_size, 1)

    q_eval = model.eval_net(embed_s).gather(1, b_a).type(torch.FloatTensor).to(device)  # (batch_size, n)
    # TODO: concat state with multiple action to calculate in parallel
    q_next = model.target_net(embed_s_).detach().type(torch.FloatTensor).to(device)
    q_target = b_r + GAMMA * q_next.max(1)[0].view(batch_size,1).type(torch.FloatTensor).to(device)
    
    loss = nn.MSELoss(q_eval, q_target)

    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

def train_network(model, project, val_version, batch_size=64):

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


    # load training data
    start, end = projects[project]
    for n in range(start, end + 1):

        if n in excluded[project] or n == val_version:
            continue
        memory = Memory(2000)   # initialize memory for each version

        loaded_data = load(project, n)
        X, y, methods, testcases, faulty_methods = loaded_data

        starting_index = 0
        num_failing_tests = int(np.sum(y == 0.))

        X = torch.tensor(binarize(X), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        N, M = X.shape

        # train for each version / each batch
        for ti in range(starting_index, num_failing_tests):
            ############ preparation ############# 
            results = {  
                # the indices of selected test cases
                "tests": [],
                # the rankings of faulty methods at each iteration
                "ranks": [],
            }
            
            spec_cache = None
            score_cache = None
            failing_idx = np.where(y == 0)[0].tolist()
            initial_tests = failing_idx[ti:ti+1]

            # for faster fault localization
            spec_cache = spectrum(X[initial_tests], y[initial_tests])
            score_cache = ochiai(*spec_cache)
            ranks = get_ranks_of_faulty_elements(methods, score_cache, faulty_methods, level=0)
            initial_min_ranks = np.min(ranks)     # get initial min rank for reward calculation
            results["tests"].append(initial_tests)
            results["ranks"].append(ranks)

            # for each test, calculate fitness function with critic, and store the transition
            while not len(results["ranks"]) > 10:
                # get current state
                tests = list(sum(results["tests"], []))
                state = model.encoder(X[tests, :])

                # calculate fitness value
                best_fitness = None
                selected_tests = None
                best_feature = None
                for i in tqdm(range(N), colour='green'):
                    if i in tests:
                        continue
                    feature = getFeature(X[tests, :], X[i])
                    fitness = model.eval_net(torch.cat([feature, state], dim=1))

                    if best_fitness is None or fitness > best_fitness:
                        best_fitness = fitness
                        best_feature = feature
                        selected_tests = [i]

                # update cache
                if selected_tests is not None:
                    e_p, n_p, e_f, n_f = spec_cache
                    for t in selected_tests:
                        if y[t] == 0:
                            # Fail
                            e_f += X[t]
                            n_f += (1 - X[t])
                        else:
                            # Pass
                            e_p += X[t]
                            n_p += (1 - X[t])
                    spec_cache = e_p, n_p, e_f, n_f
                    score_cache = ochiai(*spec_cache)

                # get rank and calculate reward
                ranks = get_ranks_of_faulty_elements(methods, score_cache, faulty_methods, level=0)
                current_min_ranks = np.min(ranks)
                reward = (initial_min_ranks - current_min_ranks) / initial_min_ranks      # relative promoted 

                # update results
                results["tests"].append(selected_tests)
                results["ranks"].append(ranks)

                # store transition
                memory.store_transition(tests, selected_tests, best_feature, reward)

                # if having a number of transitions, then do backward propagation and update the model
                if memory.memory_counter > memory.memory_size:
                    learn(X=X, model=model, memory=memory, batch_size=batch_size)


if __name__ == "__main__":
    # initialize the model
    embed_dim = 16
    model = Model(input_dim=2, embed_dim=16, hidden_dim=16)
    # encoder = Encoder(input_dim=2, hidden_dim=embed_dim)
    # critic = Critic(input_dim=2+embed_dim, hidden_dim=16, output_dim=1) # output_dim=1 because we want to output the test's "fitness" 

    train_network(model, "Chart", 1)