import numpy as np
import pandas as pd
from scipy.stats import rankdata, entropy
from enum import Enum
import torch
from scipy.spatial.distance import cdist

class TestResult(Enum):
  FAILING = 0
  SUCCESS = 1
  NEEDASK = 2
  UNKNOWN = 3

class TestCase:
  def __init__(self, origin, name, coverage, contents, oracle):
    self.origin = origin
    self.name = name
    self.coverage = coverage
    self.contents = contents
    self.oracle = oracle

  @property
  def id(self):
    return (self.origin, self.name)

class CovVecGen:
  def __init__(self, elements):
    elements = list(sorted(list(elements)))
    self.elements = elements
    self.index = {e:i for i, e in enumerate(elements)}

  def generate(self, covered):
    #if any([e not in self.elements for e in covered]):
    #  raise Exception("Unseen elements")
    vector = np.zeros(len(self.elements))
    for e in covered:
      if e in self.index:
        vector[self.index[e]] = 1.
    return vector

def ranking(l, method='max'):
    return rankdata(-np.array(l), method=method)

def matrix_to_index(X, y):
  X, y = np.array(X), np.array(y)
  assert np.all(X >= 0)
  assert np.all(np.isin(y, [0, 1]))

  e_f = np.sum(X[y==0], axis=0)
  n_f = np.sum(y == 0) - e_f
  e_p = np.sum(X[y==1], axis=0)
  n_p = np.sum(y == 1) - e_p
  return e_p, n_p, e_f, n_f

def ochiai(e_p, n_p, e_f, n_f):
  e = e_f > 0
  scores = np.zeros(e_p.shape[0])
  scores[e] = e_f[e]/np.sqrt((e_f[e]+n_f[e])*(e_f[e]+e_p[e]))
  scores[~e] = .0
  return scores

def extend_columns(a, n):
  b = np.zeros((a.shape[0], a.shape[1]+n))
  b[:,:-n] = a
  return b

def minimize_matrix(coverage_matrix, result_vector):
  is_failing_test = result_vector['value'] == TestResult.FAILING
  covered_by_failings = np.sum(coverage_matrix.values[is_failing_test, :], axis=0) > 0
  useful_tests = np.sum(np.logical_and(coverage_matrix, covered_by_failings), axis=1) > 0
  return coverage_matrix.loc[useful_tests, :], result_vector.loc[useful_tests]

def slice_data(coverage_matrix, result_vector):
  is_valid = result_vector['value'].isin([TestResult.FAILING.value, TestResult.SUCCESS.value]).values
  return coverage_matrix.loc[is_valid, :], result_vector.loc[is_valid, :], is_valid

def get_ranks(elements, scores, level=0, return_entropy=False, return_score=False, verbose=False):
  if len(elements) > 0:
    max_level = len(elements[0])
    idx = pd.MultiIndex.from_arrays(
      [[t[l] for t in elements] for l in range(max_level)],
      names=[str(l) for l in range(max_level)])
    s = pd.Series(scores, name='scores', index=idx)
    if max_level - 1  == level:
      aggr = s
    else:
      aggr = s.max(level=level)
    if verbose:
      print(aggr)
    if return_entropy and return_score:
      return aggr.index.values, ranking(aggr.values), entropy(aggr.values), aggr.values
    elif return_score:
      return aggr.index.values, ranking(aggr.values), aggr.values
    elif return_entropy:
      return aggr.index.values, ranking(aggr.values), entropy(aggr.values)
    else:
      return aggr.index.values, ranking(aggr.values)
  else:
    return []

def getState(X):
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
    # for index in target_index:
    #     method_class.append(methods[index])

    unique, indices, counts = torch.unique(X, sorted=False, return_inverse=True, return_counts=True, dim=1)
    num_ag = unique.shape[1]   # number of ambiguity group

    # impurity = 1  # calculate the gini index of the most suspicious ambiguity group
    # group_cover_min = 10
    # for j in range(unique.shape[1]):  # calculate impurity for each group
    #     group_cover_test_num = unique[:, j].sum()    # number of covered test for this group
    #     if group_cover_test_num > group_cover_min:
    #         continue

    #     group_index = torch.where(indices == j)[0].cpu().numpy().tolist()
    #     group_size = counts[j]
    #     group_cover_min = group_cover_test_num
        # group_method_class = []
        # for index in group_index:
        #     group_method_class.append(method_class[index])
        # dic = dict(Counter(group_method_class))
        # impurity = 1
        # for value in dic.values(): 
        #     impurity -= (value / group_size) * (value / group_size)
        
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
    cover = torch.tensor(1 - (cdist(X_cur[[0], :], X_all, 'jaccard'))).transpose(0,1)  # [m,1]
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
    split = torch.zeros(m,1)
    for j in range(unique.shape[1]):  # calculate values for each ambiguity group
        group_index = (indices == j)
        group_prio = torch.sum(group_index) / n  # |g|/n
        stack = torch.stack((torch.sum(X_all[:,group_index]==0, dim=1), torch.sum(X_all[:,group_index]==1, dim=1)),0) 
        # print(stack.shape)
        div = torch.min(stack, 0)[0].unsqueeze(1)
        split += div * group_prio
    return torch.cat([split, cover], dim=1).float()


