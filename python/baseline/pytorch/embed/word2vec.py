"""
Shoutout to 
https://github.com/theeluwin/pytorch-sgns/blob/master/model.py
"""
import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT

from baseline.w2v import EmbeddingsModel
# from baseline.model import create_model as 
import baseline.model as model
from baseline.utils import create_user_embed_model, load_user_embed_model

class Word2Vec(nn.Module, EmbeddingsModel):

  def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
    super(Word2Vec, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
    self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
    self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
    self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
    self.ivectors.weight.requires_grad = True
    self.ovectors.weight.requires_grad = True

  def forward(self, data):
    return self.forward_i(data)

  def forward_i(self, data):
    v = LT(data)
    v = v.cuda() if self.ivectors.weight.is_cuda else v
    return self.ivectors(v)

  def forward_o(self, data):
    v = LT(data)
    v = v.cuda() if self.ovectors.weight.is_cuda else v
    return self.ovectors(v)

  def get_dsz(self):
    return self.embedding_size

  def get_vsz(self):
    return self.vocab_size

  def lookup(self, word):
    pass

  @classmethod
  def create(cls, embeddings_set, labels, **kwargs):
    vocab_size = kwargs['vocab_size']
    embedding_size = kwargs['embed_size']

    return cls(vocab_size, embedding_size)

  @classmethod
  def load(cls, outname, **kwargs):
    model = torch.load(outname)
    return model

  def save(self, outname):
    print('saving %s' % outname)
    t.save(self, outname)


BASELINE_EMBED_MODELS = {
  'default': Word2Vec.create
}

BASELINE_EMBED_LOADERS = {
  'default': Word2Vec.load
}

def create_model(embeddings, labels, **kwargs):
  kwargs['task_fn'] = create_user_embed_model
  return model.create_model(BASELINE_EMBED_MODELS, embeddings, labels, **kwargs)

def load_model(outname, **kwargs):
  kwargs['task_fn'] = load_user_embed_model
  return model.load_model(BASELINE_EMBED_LOADERS, outname, **kwargs)
