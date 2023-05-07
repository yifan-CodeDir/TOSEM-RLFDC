import numpy as np
import torch
import argparse
import json
import os
from utils.metrics import *
from utils.d4j import load
from utils.FL import *
from tqdm import tqdm
from sklearn.preprocessing import binarize
from RL_algo.train_simple_network_more_episode import train_network, remove
from RL_algo.network import Model, getFeature, getState, getallFeature
import math

def simulate(X, y, initial_tests, method_class, spec_cache=None, score_cache=None,
             fitness_function=FDG, sbfl_formula=ochiai, weight=ochiai, model=None):
    N, M = X.shape

    tests = list(initial_tests[:])

    if spec_cache is None:
        spec_cache = spectrum(X[tests],y[tests])

    if score_cache is None:
        score_cache = sbfl_formula(*spec_cache)

    if sbfl_formula == weight:
        weights = score_cache[:]
    else:
        weights = weight(*spec_cache)

    # Optimization
    if fitness_function in [Split, FDG]:
        _, cX = torch.unique(X[tests], return_inverse=True, sorted=False, dim=1)
        cX = torch.unsqueeze(cX, 0)
    elif fitness_function == DDU_fast:
        _, cX = torch.unique(X[tests], return_inverse=True, sorted=False, dim=1)
        cX = torch.unsqueeze(cX, 0)
        unique_activities, activity_counts = torch.unique(X[tests],
            return_counts=True, sorted=False, dim=0)
    elif fitness_function == S3:
        target = torch.sum(X[tests], dim=0) > 0

    if fitness_function in [RAPTER, FLINT, EntBug]:
        comparator = lambda a, b: a < b
    else:
        comparator = lambda a, b: a > b

    best_fitness, selected_tests = None, None

    
    state = getState(X[tests, :], method_class).cuda()  
    state_embed = model.encoder(state)

    for i in tqdm(range(N), colour='green'):
        if i in tests:
            continue
        if fitness_function in [Split, FDG]:
            fitness = fitness_function(cX, X[i], w=weights)
        elif fitness_function == S3:
            fitness = fitness_function(X[tests, :], X[i], w=weights, 
                target=target, y=y[tests])
        elif fitness_function == FLINT:
            fitness = fitness_function(X[tests, :], X[i], spectrum=spec_cache, 
                y=y[tests])
        elif fitness_function == Prox:
            fitness = fitness_function(X[tests, :], X[i], y=y[tests])
        elif fitness_function == DDU_fast:
            fitness = fitness_function(X[tests, :], X[i], cX=cX, 
                unique_activities=unique_activities, 
                activity_counts=activity_counts)
        elif fitness_function in [TfD_network]:          ## calculate new metric using the trained network, need to preprocess and then feed into the network
            feature = getFeature(X[tests, :], X[i]).cuda()
            fitness = model.eval_net(torch.cat([feature, state_embed], dim=0))   
            # print(fitness)
        else:
            fitness = fitness_function(X[tests, :], X[i], w=weights)

        if best_fitness is None or comparator(fitness, best_fitness):
            # first ascent
            best_fitness = fitness
            selected_tests = [i]

    # Update Spectrum
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
        score_cache = sbfl_formula(*spec_cache)

    return selected_tests, best_fitness, spec_cache, score_cache

def simulate_network(X, y, initial_tests, method_class, spec_cache=None, score_cache=None,
             fitness_function=FDG, sbfl_formula=ochiai, weight=ochiai, model=None):
    N, M = X.shape

    tests = list(initial_tests[:])

    if spec_cache is None:
        spec_cache = spectrum(X[tests],y[tests])

    if score_cache is None:
        score_cache = sbfl_formula(*spec_cache)

    # if sbfl_formula == weight:
    #     weights = score_cache[:]
    # else:
    #     weights = weight(*spec_cache)

    # comparator = lambda a, b: a > b

    best_fitness, selected_tests = None, None

    
    state = getState(X[tests, :], method_class).cuda()  # get the state
    state_embed = model.encoder(state)

    all_feature = getallFeature(X[tests, :], X).cuda()   # get the feature in parallel（N,2）
    # feature = remove(all_feature, tests)          # （N-num_tests,2）
    state_embed_re = state_embed.repeat(N,1)  # (N,16)
    fitness = model.eval_net(torch.cat([all_feature, state_embed_re], dim=1))   # (N,1)

    fitness[tests, :] = -5 # mask fitness
    best_fitness, selected_tests = torch.max(fitness, 0)
    selected_tests = selected_tests.tolist()

    # Update Spectrum
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
        score_cache = sbfl_formula(*spec_cache)

    return selected_tests, best_fitness, spec_cache, score_cache


# python simulate_network.py --metric TfD_network --pid Lang --output ./TfD_network_output/lang.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', '-p', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default="output.json")
    parser.add_argument('--weight', '-w', type=str, default="ochiai")
    parser.add_argument('--formula', '-f', type=str, default="ochiai")
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--train', action="store_true")
    args = parser.parse_args()

    formula = eval(args.formula)
    weight = eval(args.weight)
    fitness_function = eval("TfD_network")

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
        # 'Lang':    (1, 10),
        'Chart':   (1, 26),
        'Time':    (1, 27),
        'Math':    (1, 106),
        'Closure': (1, 133)
    }

    five_fold = {
        'Lang': [0, 13, 26, 39, 52, 65],
        'Chart': [0, 5, 10, 16, 21, 26],
        'Time':  [0, 5, 11, 16, 22, 27],
        'Math':  [0, 21, 42, 64, 85, 106],
        'Closure': [0, 27, 53, 80, 106, 133]
    }

    if args.pid:
        start = args.start if args.start else projects[args.pid][0]
        end = args.end if args.end else projects[args.pid][1]
        
        projects = {
            args.pid: (start, end)
        }

    simulation_results = {}

    epoch = 30
    is_train = args.train
    
    try:
        fold = 4
        save_model_path=f"network_model_v5_fold{fold}"     
        output_path = f"./network_model_v5_fold{fold}/{args.pid}.json"

        # save_model_path=f"setting5"     
        # output_path = f"./setting5/{args.pid}.json"

        if is_train:     # if train, need to train and save model
            for p in projects:
                # initialize the model 
                model = Model(state_dim=3, action_dim=2, embed_dim=16, hidden_dim=16, cuda=True)
                start= five_fold[p][fold-1]
                end = five_fold[p][fold]     # 80% project for training, the left 20% for testing
                val_version = [i for i in range(start + 1, end + 1)]
                # start = 8   ############### need revise ##############
                # end = 1
                # model = torch.load(f"./{save_model_path}/{p}-7.pt") 
                train_network(model=model, project=p, val_version=val_version, epochs=epoch, memory_size=100, batch_size=32, save_model_path=save_model_path)
                             
        for p in projects:  # evaluate
            # load model
            
            # model = torch.load(f"./{save_model_path}/Chart-1-49.pt")  
            # model = torch.load(f"./network_model_v2/Time-1-34.pt")  
            # model = Model(state_dim=3, action_dim=2, embed_dim=16, hidden_dim=16, cuda=True)   
            start= five_fold[p][fold-1]
            end = five_fold[p][fold]     # 80% project for training, the left 20% for testing  
            model = torch.load(f"./{save_model_path}/{p}.pt") 
            # start = 5
            # end = 25

            for n in range(start + 1, end + 1):
                if n in excluded[p]:
                    # check if excluded
                    continue

                key = f"{p}-{n}"
                loaded_data = load(p, n)
                if loaded_data is None:
                    simulation_results[key] = None
                    continue
                    
                X, y, methods, testcases, faulty_methods = loaded_data
                method_class = [method[0].split("$")[0] for method in methods]
                num_failing_tests = int(np.sum(y == 0))
                print(f"[{key}] # failing tests in the test suite: {num_failing_tests}")

                # initialize
                simulation_results[key] = []
                starting_index = 0

                X = torch.tensor(binarize(X), dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)

                for ti in range(starting_index, num_failing_tests):
                    results = {
                        # the indices of selected test cases
                        "tests": [],
                        # the rankings of faulty methods at each iteration
                        "ranks": [],
                        "fitness_history": [],
                        "full_ranks": None
                    }

                    # for faster fault localization
                    spec_cache = None  # spectrum cache
                    score_cache = None # FL score cache

                    failing_idx = np.where(y == 0)[0].tolist()
                    initial_tests = failing_idx[ti:ti+1]

                    spec_cache = spectrum(X[initial_tests], y[initial_tests])
                    score_cache = formula(*spec_cache)
                    ranks = get_ranks_of_faulty_elements(methods, score_cache,
                        faulty_methods, level=0)
                    full_ranks = get_ranks_of_faulty_elements(methods,
                        formula(*spectrum(X, y)), faulty_methods, level=0)
                    results["tests"].append(initial_tests)
                    results["ranks"].append(ranks)
                    print(f"[{key}] Starting iteration with a failing test {[testcases[t] for t in initial_tests]} ({ti + 1}/{num_failing_tests}).")
                    print(f"[{key}] With the initial test, faulty method(s) {faulty_methods} is (are) ranked at {ranks}.")
                    fitness = float(fitness_function(
                        X[initial_tests], w=score_cache, cX=X[initial_tests]))
                    results["fitness_history"].append(fitness)
                    results["full_ranks"] = full_ranks

                    # for each iteration, use the network to select tests
                    while not len(results["ranks"]) > args.iter:
                        selected, fitness, spec_cache, score_cache = simulate_network(
                            X, y, sum(results["tests"], []), method_class,
                            spec_cache=spec_cache,
                            score_cache=score_cache,
                            fitness_function=fitness_function,
                            weight=weight,
                            sbfl_formula=formula,
                            model=model)

                        if selected == None:
                            break
                        
                        ranks = get_ranks_of_faulty_elements(
                            methods, score_cache, faulty_methods, level=0)
                        results["tests"].append(selected)
                        results["ranks"].append(ranks)
                        results["fitness_history"].append(float(fitness))
                        print(f"[{key}] Selected test: {[testcases[t] for t in selected]} (score: {float(fitness):.5f}). Now the faulty methods are ranked at {ranks}.")

                    simulation_results[key].append(results)

    except KeyboardInterrupt as e:
        print(e)
    finally:
        simulation_results[key].append(results)
        with open(output_path, "w") as f:
            json.dump(simulation_results, f, sort_keys=True,
                ensure_ascii=False, indent=4)
        print(f"* Results are saved to {output_path}")