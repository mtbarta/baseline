import scipy.sparse.csgraph as csg
import utils.distortions as dis
import graph_helpers as gh

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np, math
import random

class GraphRowSubSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, subsample, Z=None):
        super(GraphRowSubSampler, self).__init__()
        self.graph     = nx.to_scipy_sparse_matrix(G)
        self.n         = G.order()
        self.scale     = scale
        self.subsample = subsample
        self.val_cache = torch.DoubleTensor(self.n,subsample).zero_()
        self.idx_cache = torch.LongTensor(self.n,subsample,2).zero_()
        self.cache     = set()
        self.verbose   = False
        self.n_cached  = 0
        self.Z         = Z
        self.nbr_frac  = 0.9 # fill up this proportion of samples with neighbors
        logging.info(self)

    def __getitem__(self, index):
        if index not in self.cache:
            if self.verbose: logging.info(f"Cache miss for {index}")
            h = gh.djikstra_wrapper( (self.graph, [index]) )[0,:] if self.Z is None else self.Z[index,:]
            # add in all the edges
            cur = 0
            self.idx_cache[index,:,0] = index
            neighbors = scipy.sparse.find(self.graph[index,:])[1]
            for e in neighbors:
                self.idx_cache[index,cur,1] = int(e)
                self.val_cache[index,cur] = self.scale
                cur += 1
                if cur >= self.nbr_frac * self.subsample: break

            scratch   = np.array(range(self.n))
            np.random.shuffle(scratch)

            i = 0
            while cur < self.subsample and i < self.n:
                v = scratch[i]
                if v != index and v not in neighbors:
                    self.idx_cache[index,cur,1] = int(v)
                    self.val_cache[index,cur]   = self.scale*h[v]
                    cur += 1
                i += 1
            if self.verbose: logging.info(f"\t neighbors={neighbors} {self.idx_cache[index,:,1].numpy().T}")
            self.cache.add(index)
            self.n_cached += 1
            if self.n_cached % (max(self.n//20,1)) == 0: logging.info(f"\t Cached {self.n_cached} of {self.n}")

        # print("GraphRowSubSampler: idx shape ", self.idx_cache[index,:].size())
        return (self.idx_cache[index,:], self.val_cache[index,:])

    def __len__(self): return self.n

    def __repr__(self):
        return f"Subsample: {self.n} points with scale {self.scale} subsample={self.subsample}"


class GraphRowSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, use_cache=True):
        self.graph = nx.to_scipy_sparse_matrix(G)
        self.n     = G.order()
        self.scale = scale
        self.cache = dict() if use_cache else None

    def __getitem__(self, index):
        h = None
        if self.cache is None or index not in self.cache:
            h = gh.djikstra_wrapper( (self.graph, [index]) )
            if self.cache is not None:
                self.cache[index] = h
            #logging.info(f"info {index}")
        else:
            h = self.cache[index]
            #logging.info(f"hit {index}")

        idx = torch.LongTensor([ (index, j) for j in range(self.n) if j != index])
        v   = torch.DoubleTensor(h).view(-1)[idx[:,1]]
        return (idx, v)

    def __len__(self): return self.n

    def __repr__(self):
return f"DATA: {self.n} points with scale {self.scale}"