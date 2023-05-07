import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import time
import pandas as pd
import json
# import torch.multiprocessing as mp
# from functools import partial

sys.path.append("..")
from utils.d4j import load
from utils.FL import *
from RL_algo.network import Critic, Encoder, Model, getFeature, getallFeature, getState
from torch.nn.utils.rnn import pack_sequence
from sklearn.preprocessing import binarize
import tqdm
import matplotlib.pyplot as plt

EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 20
# TARGET_REPLACE_ITER = 10
LEARN_STEP = 5
# POOL_SIZE = 8

def remove(a, remove_index):
    rest_index = []
    for i in range(a.shape[0]):
        if i in remove_index:
            continue
        rest_index.append(i)
    rest_index = torch.tensor(rest_index).cuda()
    return torch.index_select(a, 0, rest_index)



class Memory():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        # self.pool = torch.zeros(memory_size, 4+2+1+4+11+1).cuda()      # initialize memory pool (state, action, reward)
        self.pool = torch.zeros(memory_size, 3+2+1+3+11+1).cuda()      # initialize memory pool (state, action, reward)
        self.memory_counter = 0
    
    def store_transition(self, s, a, r, s_, cur_tests, num_tests):
        transition = torch.cat([s, a, r, s_, cur_tests, num_tests],0)     # s=[current tests], a=feature of selected tests, r=relative promoted rank 
        index = self.memory_counter % self.memory_size
        self.pool[index] = transition
        self.memory_counter += 1


def learn(X, model, memory, batch_size, log_loss):
    print("Learning....")
    if model.learn_step_counter % TARGET_REPLACE_ITER == 0:   # update the param of target network during certain iterations
        model.target_net.load_state_dict(model.eval_net.state_dict())
    model.learn_step_counter += 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # random sample batch data from memory (sample according to the reward)
    # abs_reward = torch.abs(memory.pool[:, -1])
    # sample_weight = (abs_reward / torch.sum(abs_reward)).cpu()
    sample_index = torch.from_numpy(np.random.choice(np.arange(0, memory.memory_size), batch_size, replace=True)).to(device)
    # sample_index = torch.randint(0, memory.memory_size, (1, batch_size)).squeeze(0).to(device)

    batch = torch.index_select(memory.pool, 0, sample_index)
    
    # b_s = batch[:, :4]  # (batch_size, 4)
    # b_a = batch[:, 4:6] # (batch_size, 2)
    # b_r = batch[:, 6:7]  # (batch_size, 1)
    # b_s_ = batch[:, 7:11]  # (batch_size, 4)
    # b_cur_tests = batch[:, 11:22]  # (batch_size, 11)
    # b_num_tests = batch[:, 22:]

    b_s = batch[:, :3]  # (batch_size, 3)
    b_a = batch[:, 3:5] # (batch_size, 2)
    b_r = batch[:, 5:6]  # (batch_size, 1)
    b_s_ = batch[:, 6:9]  # (batch_size, 3)
    b_cur_tests = batch[:, 9:20]  # (batch_size, 11)
    b_num_tests = batch[:, 20:]

    embed_bs = model.encoder(b_s).to(device) # (batch_size, embed_dim)
    embed_bs_ = model.encoder(b_s_).to(device) # (batch_size, embed_dim)
    
    b_r = b_r.type(torch.FloatTensor).to(device) # (batch_size, 1)

    q_eval = model.eval_net(torch.cat([b_a, embed_bs], 1)).type(torch.FloatTensor).to(device)  # (batch_size, 1)

    # construct [s', a'] to calculate Q(s',a') in parallel and get q_target, need to input X to calculate action feature
    # 1. get all tests' action feature for each s'
    # 2. concat each s' with all action feature
    # 3. get the max(Q(s', a')) for each s'as q_target
    # 4. calculate loss
    # print(time.time())
    N, M = X.shape
    q_target = torch.zeros(batch_size, 1).cuda()
    for i in range(batch_size):  # process for each s'
        if b_num_tests[i] == 11:  # if the episode ends in the next state
            q_target[i] = b_r[i]
        else:
            num_tests = int(b_num_tests[i].item())
            all_a_ = torch.zeros(N-num_tests,2).cuda()
            all_s_ = embed_bs_[i].repeat(N-num_tests,1)

            cur_tests = b_cur_tests[i][:num_tests].tolist()    # get current selected tests
            
            all_a_ = getallFeature(X[cur_tests, :], X).cuda() # (m,2)
            all_a_ = remove(all_a_, cur_tests)
            # index = 0
            # for j in range(N):     
            #     if j in cur_tests:  # if j is selected, then skip
            #         continue
            #     a_ = getFeature(X[cur_tests.long(), :], X[j]).cuda() 
            #     all_a_[index] = a_
            #     index += 1
            q_next = model.target_net(torch.cat([all_a_, all_s_],1)).detach().type(torch.FloatTensor).max().to(device)  # the max q value for the next state
            q_target[i] = b_r[i] + GAMMA * q_next
    # print(time.time())
    loss = F.mse_loss(q_eval, q_target)

    log_loss.append(loss.item())

    # backward and update parameters
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

# def cal_target(b_num_tests, b_r, b_cur_tests, embed_bs_, X, model, i):
#     N, _ = X.shape
#     if b_num_tests[i] == 11:  # if the episode ends in the next state
#         return b_r[i]
#     else:
#         num_tests = int(b_num_tests[i].item())
#         all_a_ = torch.zeros(N-num_tests,2).cuda()
#         all_s_ = embed_bs_[i].repeat(N-num_tests,1)

#         cur_tests = b_cur_tests[i][:num_tests].tolist()    # get current selected tests
        
#         all_a_ = getallFeature(X[cur_tests, :], X).cuda() # (m,2)
#         all_a_ = remove(all_a_, cur_tests)
#         # index = 0
#         # for j in range(N):     
#         #     if j in cur_tests:  # if j is selected, then skip
#         #         continue
#         #     a_ = getFeature(X[cur_tests.long(), :], X[j]).cuda() 
#         #     all_a_[index] = a_
#         #     index += 1
#         q_next = model.target_net(torch.cat([all_a_, all_s_],1)).detach().type(torch.FloatTensor).max().cuda()  # the max q value for the next state
#         return b_r[i] + GAMMA * q_next


# def learn_with_multi(X, model, memory, batch_size, log_loss):
#     print("Learning....")
#     if model.learn_step_counter % TARGET_REPLACE_ITER == 0:   # update the param of target network during certain iterations
#         model.target_net.load_state_dict(model.eval_net.state_dict())
#     model.learn_step_counter += 1

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # random sample batch data from memory (sample according to the reward)
#     # abs_reward = torch.abs(memory.pool[:, -1])
#     # sample_weight = (abs_reward / torch.sum(abs_reward)).cpu()
#     sample_index = torch.from_numpy(np.random.choice(np.arange(0, memory.memory_size), batch_size, replace=True)).to(device)
#     # sample_index = torch.randint(0, memory.memory_size, (1, batch_size)).squeeze(0).to(device)

#     batch = torch.index_select(memory.pool, 0, sample_index)
    
#     b_s = batch[:, :4]  # (batch_size, 4)
#     b_a = batch[:, 4:6] # (batch_size, 2)
#     b_r = batch[:, 6:7]  # (batch_size, 1)
#     b_s_ = batch[:, 7:11]  # (batch_size, 4)
#     b_cur_tests = batch[:, 11:22]  # (batch_size, 11)
#     b_num_tests = batch[:, 22:]

#     # b_s = batch[:, :3]  # (batch_size, 3)
#     # b_a = batch[:, 3:5] # (batch_size, 2)
#     # b_r = batch[:, 5:6]  # (batch_size, 1)
#     # b_s_ = batch[:, 6:9]  # (batch_size, 3)
#     # b_cur_tests = batch[:, 9:20]  # (batch_size, 11)
#     # b_num_tests = batch[:, 20:]

#     embed_bs = model.encoder(b_s).to(device) # (batch_size, embed_dim)
#     embed_bs_ = model.encoder(b_s_).to(device) # (batch_size, embed_dim)
    
#     b_r = b_r.type(torch.FloatTensor).to(device) # (batch_size, 1)

#     q_eval = model.eval_net(torch.cat([b_a, embed_bs], 1)).type(torch.FloatTensor).to(device)  # (batch_size, 1)

#     # construct [s', a'] to calculate Q(s',a') in parallel and get q_target, need to input X to calculate action feature
#     # 1. get all tests' action feature for each s'
#     # 2. concat each s' with all action feature
#     # 3. get the max(Q(s', a')) for each s'as q_target
#     # 4. calculate loss
#     # print(time.time())
#     N, M = X.shape
#     # q_target = torch.zeros(batch_size, 1).cuda()

#     # pool = Pool(POOL_SIZE)
#     batch_idx = [i for i in range(batch_size)]  # process for each s'
#     pfunc = partial(cal_target, b_num_tests, b_r, b_cur_tests, embed_bs_, X, model)
#     q_target = mp.spawn(pfunc, nprocs=8, args=(batch_idx))
#     print(q_target)

#     # print(time.time())
#     loss = F.mse_loss(q_eval, q_target)

#     log_loss.append(loss.item())

#     # backward and update parameters
#     model.optimizer.zero_grad()
#     loss.backward()
#     model.optimizer.step()


def train_network(model, project, val_version, epochs=3, memory_size=200, batch_size=64, save_model_path="network_model_v3"):

    """
    Defects4J Simulation
    """
    excluded = {
        'Lang': [2, 23, 56],
        'Chart': [],
        'Time': [21, 22], # add Time-22 because it is useless
        'Math': [],
        'Closure': [63, 93, 105]     # add Closure-105 because it is useless
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
    start, end = projects[project]

    # read initial seed for each version
    with open("seed.json", "r") as f:
        seed_dict = json.load(f)
    # print(seed_dict)
    try:
        for n in range(start, end + 1):
            if n in excluded[project] or n in val_version:
                continue
            
            key = f"{project}-{n}"
            print(key)

            memory = Memory(memory_size)   # initialize memory for each version
            loaded_data = load(project, n)
            X, y, methods, testcases, faulty_methods = loaded_data
            method_class = [method[0].split("$")[0] for method in methods]

            starting_index = 0
            num_failing_tests = int(np.sum(y == 0.))

            X = torch.tensor(binarize(X), dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            N, M = X.shape

            # ti = starting_index   # only choose the first failing test to train
            ti = seed_dict[key]     # choose a easy-to-train failing test
            # if num_failing_tests > 1:
            #     with open("../output/FDG.json","r") as f:
            #         data_dict = json.load(f)
            #     data_dict[key][0]
            # else:
            #     ti = starting_index
            
            for epoch in range(epochs):
                # start = time.time()
                print(f"#### Epoch {epoch} ####")
                # train for each version / each batch
                # for ti in range(starting_index, num_failing_tests): # can randomly choose one to accelarate training
                
                
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
                # print(time.time())
                while not len(results["ranks"]) > 10:
                    print(len(results["ranks"]))
                    # get current state
                    tests = list(sum(results["tests"], []))
                    state = getState(X[tests, :], method_class).cuda()     
                    state_embed = model.encoder(state)    # the state_embed can be reused 
                    state_embed_re = state_embed.repeat(N,1)  # (N,16)
                    # calculate fitness value
                    # best_fitness = None
                    selected_tests = None

                    # # sample_index = np.random.choice(np.arange(0, N), 1024, replace=False).tolist()  # reduce search space
                    # for i in range(N):
                    # # for i in sample_index:
                    #     if i in tests:
                    #         continue

                    #     ########
                    #     # start_time = time.time()
                    #     feature = getFeature(X[tests, :], X[i]).cuda()  # get feature for action space
                    #     # end_time = time.time()
                    #     # print("Get feature costs: {}".format(end_time-start_time))

                    #     fitness = model.eval_net(torch.cat([feature, state_embed], dim=0)) 
                    #     # print(fitness)
                    #     if best_fitness is None or fitness > best_fitness:  # record the best test to do further exploration
                    #         best_fitness = fitness
                    #         selected_tests = [i]
                    
                    all_feature = getallFeature(X[tests, :], X).cuda()   # get the feature in parallel（N,2）
                    # feature = remove(all_feature, tests)          # （N,2）
                    fitness = model.eval_net(torch.cat([all_feature, state_embed_re], dim=1))   # (N,1)

                    fitness[tests, :] = -5 # mask fitness 
                    best_fitness, selected_tests = torch.max(fitness, 0)
                    selected_tests = selected_tests.tolist()

                    # Get the tests with the largest fitness and chose it with certain probability to continue the next iteration
                    # add random choice to explore more state
                    if np.random.uniform() >= EPSILON:
                        while True:
                            select_index = np.random.randint(0, N)
                            if select_index not in tests:   # avoid redundant test selection
                                selected_tests = [select_index]
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

                    cur_tests = torch.tensor(tests + selected_tests).cuda()    # include the selected tests
                    zeropad = torch.zeros(11-len(cur_tests)).cuda()
                    next_state = getState(X[cur_tests, :], method_class).cuda() 
                    cur_tests = torch.cat([cur_tests, zeropad])  # pad with zero
                    action = getFeature(X[tests, :], X[selected_tests[0]]).cuda()  # get feature for action

                    memory.store_transition(state, action, torch.tensor([reward]).cuda(), next_state, cur_tests, torch.tensor([len(tests)+1]).cuda())

                    # record reward
                    log_reward.append(reward)

                    # update results
                    results["tests"].append(selected_tests)
                    results["ranks"].append(ranks)              

                    # if having a number of transitions, then do backward propagation and update the model, should be learned within version, because the X may change
                    # use multi process to accelerate learning
                    if (memory.memory_counter > memory.memory_size) and (memory.memory_counter % LEARN_STEP == 0):
                        learn(X=X, model=model, memory=memory, batch_size=batch_size, log_loss=log_loss)
                    # print(time.time())

                # # saving model for each epoch
                # if (memory.memory_counter > memory.memory_size):
                #     print("Saving model...")
                #     torch.save(model, f'./{save_model_path}/{project}-{n}-{epoch}.pt')
                # end = time.time()
                # print(end-start)

            # # saving model for each project
            # print("Saving model...")
            # torch.save(model, f'./{save_model_path}/{project}-{n}.pt')

        # saving model for each fold
        print("Saving model...")
        torch.save(model, f'./{save_model_path}/{project}.pt')

    except KeyboardInterrupt as e:
        print(e)
    finally:
        dic1 = {"log_reward": log_reward}
        data = pd.DataFrame(dic1)
        data.to_csv(f"./{save_model_path}/{project}-log_reward.csv")

        dic2 = {"log_loss": log_loss}
        data = pd.DataFrame(dic2)
        data.to_csv(f"./{save_model_path}/{project}-log_loss.csv")

    # plot figure
    # print("Plotting training curve...")
    # plt.figure(figsize=(4, 4), dpi=300)

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
    