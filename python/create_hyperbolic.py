from baseline import *
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

# args.reporting = setup_reporting(**vars(args))

from baseline.pytorch.embed.corpus import GraphCorpus
from baseline.pytorch.embed.hyperbolic_models import Hyperbolic_Emb
from baseline.pytorch.embed.hyperbolic_trainer import fit
from baseline.pytorch.embed.graph_sampler import GraphRowSampler
from gensim.models.word2vec import Text8Corpus

def text8iter():
    for line in Text8Corpus('/home/matt/work/baseline/data/text8/t8'):
        yield line

def assemble_corpus():
    data_iter = text8iter()
    corpus = GraphCorpus()
    print("fitting corpus...")
    corpus.fit(data_iter)
    print("done")

    return corpus

def collate(ls):
    x, y = zip(*ls)
    return torch.cat(x), torch.cat(y)

def main(scale=True):
    batch_sz = 10
    shuffle = True

    corpus = assemble_corpus()
    graph = corpus.to_graph()

    sampler = GraphRowSampler(graph, scale)
    dl = DataLoader(sampler, batch_sz, shuffle, collate_fn=collate)

    model = Hyperbolic_Emb(order, 
                            rank, 
                            initialize=None, 
                            learn_scale=True,
                            exponential_rescale=None)

    print("training")
    trainer = fit(model, dl)


if __name__ == '__main__':
    main()