from baseline.pytorch.torchy import *
from baseline.utils import listify, revlut, get_model_file
from baseline.w2v import EmbeddingsModel
from baseline.reporting import basic_reporting
from baseline.train import Trainer, create_trainer
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor as FT
import torch as t
import time


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, embedding: EmbeddingsModel, n_negs=20, weights=None):
        super(SkipGramNegativeSampling, self).__init__()
        self.embedding = embedding
        self.vocab_size = embedding.get_vsz()
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()


class EmbeddingTrainer(Trainer):

  def __init__(self, model, **kwargs):
    super(EmbeddingTrainer, self).__init__()
    self.train_steps = 0
    self.valid_epochs = 0
    self.gpu = not bool(kwargs.get('nogpu', False))
    optim = kwargs.get('optim', 'adam')
    eta = float(kwargs.get('eta', 0.01))
    mom = float(kwargs.get('mom', 0.9))
    self.clip = float(kwargs.get('clip', 5))
    n_negs = kwargs.get('n_negs', 20)
    self.batch_size = kwargs.get('batchsz', 50)

    self.model = SkipGramNegativeSampling(embedding=model,
                                          n_negs=n_negs)
    if optim == 'adadelta':
      self.optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
    elif optim == 'adam':
      self.optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    elif optim == 'rmsprop':
      self.optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
    elif optim == 'asgd':
      self.optimizer = torch.optim.ASGD(model.parameters(), lr=eta)
    else:
      self.optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)

    if self.gpu:
      self.model = self.model.cuda()

  def _get_dims(self, ts):
    np_array = ts[0]['x']
    return np_array.shape

  def train(self, dataloader, reporting_fns):
    start_time = time.time()
    
    total_batches = int(np.ceil(len(dataloader.dataset) / self.batch_size))

    total_loss = 0.0
    for iword, owords in dataloader:
      loss = self.model(iword, owords)
      total_loss += loss.data
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      

    metrics = {}
    avg_loss = float(total_loss) / len(dataloader.dataset)
    metrics['avg_loss'] = avg_loss
    metrics['perplexity'] = np.exp(avg_loss)

    duration = time.time() - start_time
    print('Training time (%.3f sec)' % duration)

    for reporting in reporting_fns:
        reporting(metrics, self.train_epochs * self.batch_size, 'Train')
    return metrics

def fit(model, ts, vs, es, **kwargs):
  epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
  model_file = get_model_file(kwargs, 'embed', 'pytorch')

  reporting_fns = listify(kwargs.get('reporting', basic_reporting))

  after_train_fn = kwargs.get('after_train_fn', None)
  trainer = create_trainer(EmbeddingTrainer, model, **kwargs)

  for epoch in range(epochs):

    trainer.train(ts, reporting_fns)
    if after_train_fn is not None:
      after_train_fn(model)

    model.save(model_file)