from baseline import *
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
# args.reporting = setup_reporting(**vars(args))

from baseline.pytorch.embed.corpus import GraphCorpus
from baseline.pytorch.embed.hyperbolic_models import Hyperbolic_Emb
from baseline.pytorch.embed.hyperbolic_trainer import fit
from baseline.pytorch.embed.graph_sampler import GraphRowSampler
from gensim.models.word2vec import Text8Corpus

def text8iter():
    for line in Text8Corpus('/usr/local/dradmins/mbarta/baseline/data/text8'):
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
    batch_sz = 100
    shuffle = True
    rank = 15

    graph_on_disk = "text8_corpus.pkl"
    if os.path.isfile(graph_on_disk):
        corpus = GraphCorpus.load(graph_on_disk)
        graph = pickle.load(open("text8_graph.pkl", 'rb'))
    else:
        corpus = assemble_corpus()

        corpus.save(graph_on_disk)
        graph = corpus.to_graph()
        pickle.dump(graph, open("text8_graph.pkl", 'wb'))

    sampler = GraphRowSampler(graph, scale)
    dl = DataLoader(sampler, batch_sz, shuffle, collate_fn=collate)

    order = graph.order()
    model = Hyperbolic_Emb(order, 
                            rank, 
                            initialize=None, 
                            learn_scale=True,
                            exponential_rescale=None)
    model.cuda()
    print("training")
    trainer = fit(model, dl)


if __name__ == '__main__':
    main()
