import sys
sys.path.append("/root/workspace/utils") # path to find RL_algo
import torch
from model import Model
import time
import RL_algo

def find_fold(project, version):
  five_fold = {
    'Lang': [0, 13, 26, 39, 52, 65],
    'Chart': [0, 5, 10, 16, 21, 26],
    'Time':  [0, 5, 11, 16, 22, 27],
    'Math':  [0, 21, 42, 64, 85, 106],
    'Closure': [0, 27, 53, 80, 106, 133]
  }
  for fold in range(1, 6):
    start_version = five_fold[project][fold-1]
    end_version = five_fold[project][fold]
    if (version <= end_version) and (version > start_version):
      return fold

if __name__ == '__main__':
    project = "Lang"
    version = 20
    
    five_fold = {
        'Lang': [0, 13, 26, 39, 52, 65],
        'Chart': [0, 5, 10, 16, 21, 26],
        'Time':  [0, 5, 11, 16, 22, 27],
        'Math':  [0, 21, 42, 64, 85, 106],
        'Closure': [0, 27, 53, 80, 106, 133]
    }
    fold = find_fold(project, version)
    model = torch.load("/root/workspace/model/network_model_v5_fold{}/{}.pt".format(fold, project), map_location=torch.device('cpu'))
    state = []
    feature = []
    # get state
    for i in range(1, 4):
        state.append(float(sys.argv[i]))

    # get feature 
    for i in range(4, 6):
        feature.append(float(sys.argv[i]))

    state = torch.tensor(state).float()
    feature = torch.tensor(feature).float()

    state_embed = model.encoder(state)
    fitness = model.eval_net(torch.cat([feature, state_embed], dim=0))   # (N,1)
    print(fitness.item())
    # print(-10000)


