import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import time
import pandas as pd

sys.path.append("..")
from utils.d4j import load
from utils.FL import *
from RL_algo.network import Critic, Encoder, Model, getFeature, getState
from torch.nn.utils.rnn import pack_sequence
from sklearn.preprocessing import binarize
import tqdm
import matplotlib.pyplot as plt

EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
    

class Memory():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.pool = torch.zeros(memory_size, 4+2+1+4+10+1).cuda()      # initialize memory pool (state, action, reward)
        self.memory_counter = 0
    
    def store_transition(self, s, a, r, s_, cur_tests, num_tests):
        transition = torch.cat([s, a, r, s_, cur_tests, num_tests],0)     # s=[current tests], a=feature of selected tests, r=relative promoted rank 
        index = self.memory_counter % self.memory_size
        self.pool[index] = transition
        self.memory_counter += 1


def learn(X, model, memory, batch_size, log_loss):
    if model.learn_step_counter % TARGET_REPLACE_ITER == 0:   # update the param of target network during certain iterations
        model.target_net.load_state_dict(model.eval_net.state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # random sample batch data from memory (sample according to the reward)
    # abs_reward = torch.abs(memory.pool[:, -1])
    # sample_weight = (abs_reward / torch.sum(abs_reward)).cpu()
    sample_index = torch.from_numpy(np.random.choice(np.arange(0, memory.memory_size), batch_size, replace=False)).to(device)
    # sample_index = torch.randint(0, memory.memory_size, (1, batch_size)).squeeze(0).to(device)

    batch = torch.index_select(memory.pool, 0, sample_index)
    
    b_s = batch[:, :4]  # (batch_size, 4)
    b_a = batch[:, 4:6] # (batch_size, 2)
    b_r = batch[:, 6:7]  # (batch_size, 1)
    b_s_ = batch[:, 7:11]  # (batch_size, 4)
    b_cur_tests = batch[:, 11:21]  # (batch_size, 10)
    b_num_tests = batch[:, 21:]

    embed_bs = model.encoder(b_s).to(device) # (batch_size, embed_dim)
    embed_bs_ = model.encoder(b_s_).to(device) # (batch_size, embed_dim)
    
    b_r = b_r.type(torch.FloatTensor).to(device) # (batch_size, 1)

    q_eval = model.eval_net(torch.cat([embed_bs, b_a], 1)).type(torch.FloatTensor).to(device)  # (batch_size, 1)
    # construct [s', a'] to calculate Q(s',a') in parallel and get q_target, need to input X to calculate action feature
    # 1. get all tests' action feature for each s'
    # 2. concat each s' with all action feature
    # 3. get the max(Q(s', a')) for each s'as q_target
    # 4. calculate loss
    N, M = X.shape
    q_target = torch.zeros(batch_size, 1)
    for i in range(batch_size):  # process for each s'
        if b_num_tests[i] == 10:  # if the episode ends in the next state
            q_target[i] = b_r[i]
        else:
            all_a_ = torch.zeros(N-b_num_tests[i],2)
            all_s_ = embed_bs_[i].repeat(N-b_num_tests[i],1)

            cur_tests = b_cur_tests[i][:b_num_tests[i]]    # get current selected tests
            index = 0
            for j in range(N):     
                if j in cur_tests:  # if j is selected, then skip
                    continue
                a_ = getFeature(X[cur_tests, :], X[j]).cuda() 
                all_a_[index] = a_
                index += 1
            q_next = model.target_net(torch.cat([all_s_,all_a_],1)).detach().type(torch.FloatTensor).max().to(device)  # the max q value for the next state
            q_target[i] = b_r[i] + GAMMA * q_next.type(torch.FloatTensor).to(device)

    loss = F.mse_loss(q_eval, q_target)

    log_loss.append(loss.item())

    # backward and update parameters
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()



def train_network(model, project, start_version, end_version, epochs=3, memory_size=200, batch_size=64):

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
        # 'Lang':    (1, 5),
        'Chart':   (1, 26),
        'Time':    (1, 27),
        'Math':    (1, 106),
        'Closure': (1, 133)
    }

    print("########### Training ############")
    # load training data
    # start, end = projects[project]
    log_reward = []    # initialize log list
    log_loss = []
    for epoch in range(epochs):

        print(f"#### Epoch {epoch} ####")

        for n in range(start_version, end_version+1):

            if n in excluded[project]:
                continue
                
            print(f"{project}-{n}")

            memory = Memory(memory_size)   # initialize memory for each version

            loaded_data = load(project, n)
            X, y, methods, testcases, faulty_methods = loaded_data
            method_class = [method[0].split("$")[0] for method in methods]

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
                    print(len(results["ranks"]))
                    # get current state
                    tests = list(sum(results["tests"], []))
                    num_tests = len(tests)     # get the current number of tests
                    state = getState(X[tests, :], method_class).cuda()     
                    state_embed = model.encoder(state)    # the state_embed can be reused 

                    # calculate fitness value
                    best_fitness = None
                    selected_tests = None
                    best_test = None

                    # sample_index = np.random.choice(np.arange(0, N), 1024, replace=False).tolist()  # reduce search space
                    for i in range(N):
                    # for i in sample_index:
                        if i in tests:
                            continue
                        
                        # initialize temp cache to try with each candidate test
                        temp_spec_cache = copy.deepcopy(spec_cache)   # deep copy to only use value
                        # temp_score_cache = score_cache   # deep copy to only use value
                        
                        ########
                        # start_time = time.time()
                        feature = getFeature(X[tests, :], X[i]).cuda()  # get feature for action space
                        # end_time = time.time()
                        # print("Get feature costs: {}".format(end_time-start_time))
                        fitness = model.eval_net(torch.cat([feature, state_embed], dim=0))
                        selected_tests = [i]   
                        
                        if best_fitness is None or fitness > best_fitness:  # record the best test to do further exploration
                            best_fitness = fitness
                            best_test = i

                        # use feature and fitness of all candidate tests for training
                        # update temp cache to calculate ranks
                        if selected_tests is not None:
                            e_p, n_p, e_f, n_f = temp_spec_cache
                            for t in selected_tests:
                                if y[t] == 0:
                                    # Fail
                                    e_f += X[t]
                                    n_f += (1 - X[t])
                                else:
                                    # Pass
                                    e_p += X[t]
                                    n_p += (1 - X[t])
                            # spec_cache = e_p, n_p, e_f, n_f
                            temp_score_cache = ochiai(*temp_spec_cache)

                        # get rank and calculate reward
                        # start_time = time.time()
                        ranks = get_ranks_of_faulty_elements(methods, temp_score_cache, faulty_methods, level=0)
                        # end_time = time.time()
                        # print("Get rank costs: {}".format(end_time-start_time))
                        current_min_ranks = np.min(ranks)  
                        reward = torch.tensor([(initial_min_ranks - current_min_ranks) / initial_min_ranks]).cuda()      # relative promoted
                        # print(reward)
                        # store transition
                        cur_tests = torch.tensor(tests + [i]).cuda()
                        zeropad = torch.zeros(11-len(cur_tests)).cuda()
                        next_state = getState(X[cur_tests, :], method_class).cuda() 
                        cur_tests = torch.cat([cur_tests, zeropad])  # pad with zero
                        memory.store_transition(state, feature, reward, next_state, cur_tests, num_tests)

                    # Get the tests with the largest fitness and chose it with certain probability to continue the next iteration
                    # add random choice to explore more state
                    if np.random.uniform() >= EPSILON:
                        while True:
                            select_index = np.random.randint(0, N)
                            if select_index not in tests:   # avoid redundant test selection
                                selected_tests = [best_test]
                                break 

                    # update cache using the real selected test
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

                    ranks = get_ranks_of_faulty_elements(methods, score_cache, faulty_methods, level=0)
                    current_min_ranks = np.min(ranks)  
                    reward = (initial_min_ranks - current_min_ranks) / initial_min_ranks
                    # record reward
                    log_reward.append(reward)

                    # update results
                    results["tests"].append(selected_tests)
                    results["ranks"].append(ranks)              

                    # if having a number of transitions, then do backward propagation and update the model, should be learned within version, because the X may change
                    if (memory.memory_counter > memory.memory_size):
                        learn(X=X, model=model, memory=memory, batch_size=batch_size, log_loss=log_loss)


            # saving model for each version
            print("Saving model...")
            torch.save(model, f'./network_model/{project}-{epoch}-{n}.pt')

        # saving model for each epoch
        print("Saving model...")
        torch.save(model, f'./network_model/{project}-{epoch}.pt')


    # plot figure
    print("Plotting training curve...")
    plt.figure(figsize=(4, 4), dpi=300)

    dic1 = {"log_reward": log_reward}
    data = pd.DataFrame(dic1)
    data.to_csv("log_reward.csv")

    dic2 = {"log_loss": log_loss}
    data = pd.DataFrame(dic2)
    data.to_csv("log_loss.csv")

    # x_reward = range(len(log_reward))
    # l_reward,  = plt.plot(x_reward, log_reward)
    # x_loss = range(len(log_reward))
    # l_loss,  = plt.plot(x_loss, log_loss)

    # plt.legend(handles=[l_reward, l_loss], labels=['reward', 'loss'])
    # plt.savefig(f'{project}_{start_version}-{end_version}.pdf')


if __name__ == "__main__":
    # initialize the model
    embed_dim = 16
    model = Model(input_dim=2, embed_dim=16, hidden_dim=16)
    # encoder = Encoder(input_dim=2, hidden_dim=embed_dim)
    # critic = Critic(input_dim=2+embed_dim, hidden_dim=16, output_dim=1) # output_dim=1 because we want to output the test's "fitness" 

    train_network(model, "Chart", 1)
    