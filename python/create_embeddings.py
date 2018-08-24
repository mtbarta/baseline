import numpy as np
import codecs
import argparse
from baseline import *  #ugh
from baseline.pytorch.embed.word2vec import Word2Vec, create_model
from baseline.pytorch.embed.train import fit
from baseline.pytorch.embed.preprocessor import Preprocessor
from torch.utils.data import Dataset, DataLoader
from os import sys, path, makedirs
import pickle

parser = argparse.ArgumentParser(description='Train a text classifier')
parser.add_argument('--visdom', help='Turn on visdom reporting', type=str2bool, default=False)
parser.add_argument('--tensorboard', help='Turn on tensorboard reporting', type=str2bool, default=False)
parser.add_argument('--eta', help='Initial learning rate', default=0.01, type=float)
parser.add_argument('--mom', help='SGD Momentum', default=0.9, type=float)
parser.add_argument('--start_decay_epoch', type=int, help='At what epoch should we start decaying')
parser.add_argument('--decay_rate', default=0.0, type=float, help='Learning rate decay')
parser.add_argument('--decay_type', help='What learning rate decay schedule')
parser.add_argument('--dir', help='data directory', required=True)
parser.add_argument('--file', help='raw training data', required=True)
parser.add_argument('--vocab_size', help='vocab size', required=False, default=20000)
parser.add_argument('--embed_size', help='embed', required=False, default=100)


parser.add_argument('--save', help='Save basename', default='classify_sentence_pytorch')
parser.add_argument('--nogpu', help='Do not use GPU', default=True)
parser.add_argument('--optim', help='Optim method', default='adam', choices=['adam', 'adagrad', 'adadelta', 'sgd', 'asgd'])
parser.add_argument('--unif', help='Initializer bounds for embeddings', default=0.25)
parser.add_argument('--epochs', help='Number of epochs', default=25, type=int)
parser.add_argument('--batchsz', help='Batch size', default=50, type=int)
parser.add_argument('--clean', help='Do cleaning', action='store_true', default=True)
parser.add_argument('--outfile', help='Output file base', default='./classify-model')
parser.add_argument('--backend', help='Which deep learning framework to use', default='pytorch')
parser.add_argument('--keep_unused', help='Keep unused vocabulary terms as word vectors', default=False, type=str2bool)
parser.add_argument('--model_type', help='Name of model to load and train', default='default')
parser.add_argument('--bounds', type=int, default=16000, help='Tell optim decay functionality how many steps before applying decay')

args = parser.parse_args()


from baseline.pytorch import long_0_tensor_alloc as vec_alloc
from baseline.pytorch import tensor_reverse_2nd as rev2nd


class PermutedSubsampledCorpus(Dataset):
  def __init__(self, datapath, ws=None):
    data = pickle.load(open(datapath, 'rb'))
    if ws is not None:
      self.data = []
      for iword, owords in data:
        if random.random() > ws[iword]:
          self.data.append((iword, owords))
    else:
      self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    iword, owords = self.data[idx]
    return iword, np.array(owords)

args.reporting = setup_reporting(**vars(args))

# clean_fn = TSVSeqLabelReader.do_clean if args.clean else None
# src_vec_trans = rev2nd if args.rev else None

# print(clean_fn, src_vec_trans)
# reader = create_pred_reader(args.mxlen, zeropadding, clean_fn, vec_alloc, src_vec_trans)
# vocab, labels = reader.build_vocab([args.train, args.test, args.valid])
# unif = 0 if args.static else args.unif
preproc = Preprocessor(data_dir=args.dir)

preproc.build(path.join(args.dir, args.file))
preproc.convert(path.join(args.dir, args.file))

dataset = PermutedSubsampledCorpus(path.join(args.dir, 'train.dat'))
dataset_size = len(dataset.data)
dataloader = DataLoader(dataset, batch_size=args.batchsz, shuffle=True)
    
# EmbeddingsModelType = GloVeModel if args.embed.endswith(".txt") else Word2VecModel
# embeddings = {}
# embeddings['word'] = EmbeddingsModelType(args.embed, vocab, unif_weight=args.unif, keep_unused=args.keep_unused)
# feature2index = {}
# feature2index['word'] = embeddings['word'].vocab

# ts = reader.load(args.train, feature2index, args.batchsz, shuffle=True)
# print('Loaded training data')

# vs = reader.load(args.valid, feature2index, args.batchsz)
# print('Loaded valid data')

# es = reader.load(args.test, feature2index, 2)
# print('Loaded test data')
# print('Number of labels found: [%d]' % len(labels))
arguments = vars(args)
model = create_model(None,None, 
                    task_type='embed', **arguments)

fit(model, dataloader, None, None, **arguments)
