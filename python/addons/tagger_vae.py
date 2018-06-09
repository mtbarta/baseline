import torch
import torch.nn as nn
import math
import json
from baseline.model import Tagger, create_tagger_model, load_tagger_model
from baseline.pytorch.torchy import (pytorch_embedding, 
                                    pytorch_rnn, 
                                    pytorch_conv1d, 
                                    pytorch_activation,
                                    pytorch_linear,
                                    pytorch_lstm,
                                    append2seq)
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

cudnn.benchmark = True

class Encoder(nn.Module):
  def __init__(self, input_sz, hidden_sz, vae_sz, pdrop):
    super(Encoder, self).__init__()
    self.rnn, _ = pytorch_lstm(input_sz, hidden_sz, 'bilstm', 1, pdrop)

    self.mu = pytorch_linear(hidden_sz*2, vae_sz)
    self.epsilon = pytorch_linear(hidden_sz*2, vae_sz)

  def forward(self, input):
    """
    this assumes that the input is a packed sequence.
    """
    encoded, hidden = self.rnn(input)
    encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded)
    mu = self.mu(encoded)
    epsilon = self.epsilon(encoded)

    return mu, epsilon

class Decoder(nn.Module):
  def __init__(self, input_sz, hidden_sz, output_sz, pdrop):
    super(Decoder, self).__init__()
    self.rnn, _ = pytorch_lstm(input_sz, hidden_sz, 'bilstm', 1, pdrop)

    self.output = pytorch_linear(hidden_sz*2, output_sz)

  def forward(self, input):
    """
    this assumes that the input is a packed sequence.
    """
    decoded, hidden = self.rnn(input)
    # encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded)
    # mu = self.mu(encoded)
    # epsilon = self.epsilon(encoded)

    # return mu, epsilon
    return decoded

class BetaVariationalAutoEncoder(nn.Module, Tagger):
  """
  creates a variational autoencoder as defined by Kingma and Welling.

  refer to https://github.com/pytorch/examples/blob/master/vae/main.py

  :see
  https://arxiv.org/abs/1312.6114
  and
  https://arxiv.org/abs/1401.4082

  :tutorial
  https://arxiv.org/abs/1606.05908
  """
  def __init__(self):
    super(BetaVariationalAutoEncoder, self).__init__()
    torch.manual_seed(12345)

  def to_gpu(self):
    self.gpu = False
    # self.cuda()
    # self.crit.cuda()
    return self

  @classmethod
  def load(cls, outname, **kwargs):
      model = torch.load(outname)
      return model

  def save(self, outname):
      print('saving %s' % outname)
      torch.save(self, outname)

  @classmethod
  def create(cls, embeddings, labels, **kwargs):
    word_vec = embeddings['word']
    char_vec = embeddings['char']

    finetune = kwargs.get('finetune', True)
    filtsz = kwargs.get('cfiltsz')
    pdrop = float(kwargs.get('dropout', 0.5))
    hsz = kwargs.get('hsz')
    mu_sz = kwargs.get("mu_sz")
    wsz = kwargs.get("wsz")
    activation_type = kwargs.get('activation', 'relu')
    
    char_dsz = char_vec.dsz

    model = cls()
    model.activation_type = activation_type
    model.pdrop = pdrop
    model.pdropin_value = float(kwargs.get('dropin', 0.0))

    model.labels = labels

    model.beta = kwargs.get('beta')
    
    model._char_word_conv_embeddings(filtsz, char_dsz, wsz, pdrop)
    model._init_embeddings(word_vec, char_vec)

    model.dropout = nn.Dropout(pdrop)

    # model.encoder = model._init_encoder(hsz, mu_sz, pdrop)
    # model.mu = pytorch_linear(hsz*2, mu_sz)
    # model.epsilon = pytorch_linear(hsz*2, mu_sz)
    model.encoder = Encoder(model.wchsz + model.word_dsz, hsz, mu_sz, pdrop)

    # model.decoder = model._init_decoder(hsz, mu_sz, pdrop)
    model.decoder = Decoder(mu_sz, hsz, len(model.word_vocab), pdrop)

    # model.output = pytorch_linear(mu_sz*2, len(model.word_vocab))

    return model

  def _init_embeddings(self, word_vec, char_vec):
    """
    initialize the word and character embeddings of the model.

    :returns void
    """
    if word_vec is not None:
      self.word_vocab = word_vec.vocab
      self.wembed = pytorch_embedding(word_vec)
      self.word_dsz = word_vec.dsz

    self.char_vocab = char_vec.vocab
    self.cembed = pytorch_embedding(char_vec)
      
  def _char_word_conv_embeddings(self, filtsz, char_dsz, wchsz, pdrop):
    self.char_convs = []
    for fsz in filtsz:
        pad = fsz//2
        conv = nn.Sequential(
            pytorch_conv1d(char_dsz, wchsz, fsz, padding=pad),
            pytorch_activation(self.activation_type)
        )
        self.char_convs.append(conv)
        # Add the module so its managed correctly
        self.add_module('char-conv-%d' % fsz, conv)

    # Width of concat of parallel convs
    self.wchsz = wchsz * len(filtsz)
    self.word_ch_embed = nn.Sequential()
    append2seq(self.word_ch_embed, (
        #nn.Dropout(pdrop),
        pytorch_linear(self.wchsz, self.wchsz),
        pytorch_activation(self.activation_type)
    ))

  def create_sentence_representation(self, x, xch, seqlen, batchsz):
    words_over_time = self.char2word(xch.view(seqlen * batchsz, -1)).view(seqlen, batchsz, -1)

    if x is not None:
      word_vectors = self.wembed(x)
      words_over_time = torch.cat([words_over_time, word_vectors], 2)

    return words_over_time

  def encode(self, x, xch, lengths):
    batchsz = xch.size(1)
    seqlen = xch.size(0)

    words_over_time = self.create_sentence_representation(x, xch, seqlen, batchsz)
    
    dropped = self.dropout(words_over_time)
    # output = (T, B, H)

    packed = torch.nn.utils.rnn.pack_padded_sequence(dropped, lengths.tolist())

    # encoded, hidden = self.encoder(packed)
    # encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded)
    # mu = self.mu(encoded)
    # epsilon = self.epsilon(encoded)
    mu, epsilon = self.encoder(packed)

    return mu, epsilon

  def decode(self, input):
    out = self.decoder(input)
    return out
    # return out.view(out.size(0), out.size(1), -1).transpose(0, 1).contiguous()

  def compute_loss(self, inputs):
    x = inputs[0].transpose(0, 1).contiguous()
    xch = inputs[1].transpose(0, 1).contiguous()
    lengths = inputs[2]
    tags = inputs[3]
    preds, mu, logvar = self._forward(x, xch, lengths)

    return self.loss_func(preds, tags, mu, logvar)


  def loss_func(self, preds, tags, mu, logvar):
    print(preds.size())
    # preds = preds.transpose(1, 2)
    # print(preds.size())
    cross_entropy = F.cross_entropy(preds, tags)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return self.beta * KLD + cross_entropy

  def forward(self, input):
    x = input[0].transpose(0, 1).contiguous()
    xch = input[1].transpose(0, 1).contiguous()
    lengths = input[2]

    return self._forward(x, xch, lengths)

  def _forward(self, x, xch, lengths):
    mu, logvar = self.encode(x, xch, lengths)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
    

  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu


  def make_input(self, batch_dict):
    x = batch_dict['x']
    xch = batch_dict['xch']
    y = batch_dict.get('y', None)
    lengths = batch_dict['lengths']
    ids = batch_dict.get('ids', None)

    if self.training and self.pdropin_value > 0.0:
        UNK = self.word_vocab['<UNK>']
        PAD = self.word_vocab['<PAD>']
        mask_pad = x != PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(self.pdropin_value).byte()
        x.masked_fill_(mask_pad & mask_drop, UNK)

    lengths, perm_idx = lengths.sort(0, descending=True)
    x = x[perm_idx]
    xch = xch[perm_idx]
    if y is not None:
        y = y[perm_idx]

    if ids is not None:
        ids = ids[perm_idx]

    if self.gpu:
        x = x.cuda()
        xch = xch.cuda()
        if y is not None:
            y = y.cuda()

    if y is not None:
        y = torch.autograd.Variable(y.contiguous())

    return torch.autograd.Variable(x), torch.autograd.Variable(xch), lengths, y, ids

  def char2word(self, xch_i):
    # For starters we need to perform embeddings for each character
    # (TxB) x W -> (TxB) x W x D
    char_embeds = self.cembed(xch_i)
    # (TxB) x D x W
    char_vecs = char_embeds.transpose(1, 2).contiguous()
    mots = []
    for conv in self.char_convs:
        # In Conv1d, data BxCxT, max over time
        mot, _ = conv(char_vecs).max(2)
        mots.append(mot)

    mots = torch.cat(mots, 1)
    output = self.word_ch_embed(mots)
    return mots + output

  def predict(self, batch_dict):
      return classify_bt(self, batch_dict['x'])

  def get_labels(self):
      return self.labels

  def get_vocab(self):
      return self.vocab

def create_model(labels, embeddings, **kwargs):
    tagger = BetaVariationalAutoEncoder.create(embeddings, labels, **kwargs)
    return tagger


def load_model(modelname, **kwargs):
    return BetaVariationalAutoEncoder.load(modelname, **kwargs)