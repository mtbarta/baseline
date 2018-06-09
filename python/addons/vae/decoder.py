import torch.nn as nn
import torch.nn.functional as F
from baseline.pytorch.torchy import (pytorch_lstm,
                                     pytorch_linear)

class Decoder(nn.Module):
  def __init__(self, input_sz, hidden_sz, output_sz, pdrop):
    self.rnn = pytorch_lstm(input_sz, hidden_sz, 'bilstm', 1, pdrop)

    self.output = pytorch_linear(hidden_sz*2, output_sz)

  def forward(self, input):
    """
    this assumes that the input is a packed sequence.
    """
    decoded, hidden = self.rnn(input)
    # encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded)
    mu = self.mu(encoded)
    epsilon = self.epsilon(encoded)

    return mu, epsilon