from baseline.pytorch.torchy import *
from baseline.utils import listify, revlut, get_model_file
from baseline.w2v import EmbeddingsModel
from baseline.reporting import basic_reporting
from baseline.train import Trainer, create_trainer
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor as FT
from baseline.pytorch.embed.hyperbolic_optimizer import SVRG
from baseline.pytorch.embed.hyperbolic_models import Hyperbolic_Emb
import torch as t
import time


def cu_var(x, requires_grad=None, volatile=False):
    if isinstance(x, list):
        return [cu_var(u, requires_grad=requires_grad, volatile=volatile) for u in x]
    if isinstance(x, tuple):
        return tuple([cu_var(u, requires_grad, volatile) for u in list(x)])
    #rg = not volatile if requires_grad is None else requires_grad
    #vx = Variable(x, requires_grad=rg, volatile=volatile)
    return x.cuda() if torch.cuda.is_available() else x
    return x

class EmbeddingTrainer(Trainer):

  def __init__(self, model, **kwargs):
    super(EmbeddingTrainer, self).__init__()
    self.train_steps = 0
    self.valid_epochs = 0
    self.gpu = not bool(kwargs.get('nogpu', False))
    optim = kwargs.get('optim', 'adam')
    self.eta = float(kwargs.get('eta', 0.01))
    mom = float(kwargs.get('mom', 0.9))
    self.clip = float(kwargs.get('clip', 5))
    n_negs = kwargs.get('n_negs', 20)
    self.batch_size = kwargs.get('batchsz', 50)
    self.model = model
    # order = kwargs.get('order', None)
    # rank = kwargs.get('rank', None)


  def _get_dims(self, ts):
    np_array = ts[0]['x']
    return np_array.shape

  def train(self, dataloader, reporting_fns):
    start_time = time.time()

    base_opt = torch.optim.Adagrad
    self.optimizer = SVRG(self.model.parameters(),
                          lr=self.eta, T=10, 
                          data_loader=dataloader,
                          opt=base_opt)
    
    total_batches = int(np.ceil(len(dataloader.dataset) / self.batch_size))

    l = 0.0
    self.model.train(True)
    for data in dataloader:
      def closure(data=data, target=None):
        _data = data if target is None else (data,target)
        c = self.model.loss(cu_var(_data))
        c.backward()
        return c.data[0]
      l += self.optimizer.step(closure)

      # Projection
      m.normalize()
      

    metrics = {}
    avg_loss = float(l) / len(dataloader.dataset)
    metrics['avg_loss'] = avg_loss
    metrics['perplexity'] = np.exp(avg_loss)

    duration = time.time() - start_time
    print('Training time (%.3f sec)' % duration)

    for reporting in reporting_fns:
        reporting(metrics, self.train_epochs * self.batch_size, 'Train')
    return metrics

def fit(model, ts, **kwargs):
  epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
  model_file = get_model_file(kwargs, 'embed', 'pytorch')

  reporting_fns = listify(kwargs.get('reporting', basic_reporting))

  after_train_fn = kwargs.get('after_train_fn', None)

  trainer = create_trainer(EmbeddingTrainer, model, **kwargs)

  for epoch in range(epochs):

    trainer.train(ts, reporting_fns)
    print("epoch: ", epoch)
    if after_train_fn is not None:
      after_train_fn(model)

    model.save(model_file)